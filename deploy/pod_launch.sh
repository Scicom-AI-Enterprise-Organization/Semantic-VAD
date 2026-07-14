#!/usr/bin/env bash
# Full-scale launcher. Runs ON the pod. Phase 1 multilingual (parallel), Phase 2 Malaysian
# (per-subset: predownload zips once, then parallel sharded workers). Uploads mp3 shards to HF.
set -uo pipefail
export PATH=/root/venv/bin:/root/.local/bin:$PATH
set -a; . /root/.hf_env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_XET_HIGH_PERFORMANCE=1
export SVAD_REPO="${SVAD_REPO:-Scicom-intl/semantic-vad-eot}"
export AUDIO_FORMAT=mp3
export SHARD_ROWS="${SHARD_ROWS:-20000}"
ML_LIMIT="${ML_LIMIT:-500000}"
MS_LIMIT="${MS_LIMIT:-500000}"
N_ZIPS="${N_ZIPS:-6}"
CONC="${CONC:-8}"
cd /root/semantic-vad
L=/root/scale; mkdir -p "$L"

echo "=== clean repo data/ ==="
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["SVAD_REPO"]
api.create_repo(repo, repo_type="dataset", exist_ok=True)
for f in list(api.list_repo_files(repo, repo_type="dataset")):
    if f.startswith("data/") and f.endswith(".parquet"):
        api.delete_file(f, repo, repo_type="dataset"); print("removed", f, flush=True)
PY

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done; }
# per-worker timeout backstop: a transient HF stall can't block a whole phase's `wait`.
launch(){ local name=$1 secs=$2; shift 2; env "$@" timeout "${secs}s" python /root/pod_scale_big.py > "$L/$name.log" 2>&1 & }

echo "=== PHASE1 multilingual (limit=$ML_LIMIT each) ==="
for lang in english french german italian japanese korean mandarin polish portuguese russian spanish thai turkish; do
  throttle
  launch "ml-$lang" 14400 KIND=ml CONFIG="$lang" LIMIT="$ML_LIMIT"
  echo "launched ml-$lang"
done
wait
echo "PHASE1_DONE"

echo "=== PHASE2 malaysian (limit=$MS_LIMIT each) ==="
for sub in dialects imda malaysian parliament science_english; do
  wd=/root/data/zips-$sub
  echo "predownload + probe $sub -> $wd"
  # Predownload zips once (shared) and report the config's file count so we can size shards
  # to it -- more shards than files would leave empty shards and underproduce.
  info=$(ZIP_WORKDIR=$wd N_ZIPS=$N_ZIPS SUB=$sub python - <<'PY'
import os
from datasets import load_dataset
from semantic_vad.malaysian_audio import discover_zip_names, zip_prefix, DownloadZipResolver
sub=os.environ["SUB"]; tok=os.environ["HF_TOKEN"]; wd=os.environ["ZIP_WORKDIR"]; n=int(os.environ["N_ZIPS"])
zips=discover_zip_names(zip_prefix(sub, "streaming"), token=tok)[:n]
r=DownloadZipResolver(zips, token=tok, in_ram=False, workdir=wd)
ds=load_dataset("malaysia-ai/Malaysian-STT", sub, split="train", streaming=True)
print("MEMBERS", len(r.available_members()), "NSHARDS", ds.n_shards, flush=True)
PY
  )
  echo "  $info"
  nfiles=$(echo "$info" | grep -oE "NSHARDS [0-9]+" | awk '{print $2}'); nfiles=${nfiles:-1}
  shards=$(( CONC < nfiles ? CONC : nfiles )); [ "$shards" -lt 1 ] && shards=1
  per=$(( (MS_LIMIT + shards - 1) / shards ))
  echo "  -> $shards shards x $per rows ($sub)"
  for j in $(seq 0 $((shards-1))); do
    launch "ms-$sub-s$j" 14400 KIND=ms CONFIG="$sub" LIMIT="$per" \
      SHARD_IDX=$j SHARD_CNT=$shards ZIP_WORKDIR=$wd N_ZIPS=$N_ZIPS
  done
  wait
  echo "MS_SUBSET_DONE $sub"
  rm -rf "$wd"
done
echo "PHASE2_DONE"

python /root/pod_finalize.py || true
echo "ALL_DONE"
