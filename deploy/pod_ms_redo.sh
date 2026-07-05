#!/usr/bin/env bash
# Re-run Malaysian only: whole-recording audio + sentence-punctuation turns (fixes the
# mid-sentence "eot" from ASR-segment turns). Deletes old ms shards, rebuilds, re-splits.
set -uo pipefail
export PATH=/root/venv/bin:/root/.local/bin:$PATH
set -a; . /root/.hf_env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_XET_HIGH_PERFORMANCE=1
export SVAD_REPO=Scicom-intl/semantic-vad-eot AUDIO_FORMAT=mp3 SHARD_ROWS=20000
export MS_MODE=whole MS_TURN_MODE=sentence TURN_GAP=1.5
MS_LIMIT="${MS_LIMIT:-500000}"; N_ZIPS="${N_ZIPS:-4}"; CONC="${CONC:-30}"
cd /root/semantic-vad
L=/root/scale; mkdir -p "$L"

echo "=== delete old ms_* shards (all splits) ==="
python - <<'PY'
import os
from huggingface_hub import HfApi, CommitOperationDelete
api=HfApi(token=os.environ["HF_TOKEN"]); repo=os.environ["SVAD_REPO"]
dels=[f for f in api.list_repo_files(repo, repo_type="dataset")
      if f.endswith(".parquet") and f.split("/")[-1].startswith("ms-")]
print("deleting", len(dels), "old ms files", flush=True)
for i in range(0, len(dels), 64):
    api.create_commit(repo, repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in dels[i:i+64]],
        commit_message="remove old segment-based ms shards")
PY

launch(){ local name=$1 secs=$2; shift 2; env "$@" timeout "${secs}s" python /root/pod_scale_big.py > "$L/$name.log" 2>&1 & }

for sub in dialects imda malaysian parliament science_english; do
  wd=/root/data/zips-$sub
  echo "predownload $sub (whole zips), 15min cap"
  info=$(ZIP_WORKDIR=$wd N_ZIPS=$N_ZIPS SUB=$sub timeout 900 python - <<'PY'
import os
from datasets import load_dataset
from semantic_vad.malaysian_audio import discover_zip_names, zip_prefix, DownloadZipResolver
sub=os.environ["SUB"]; tok=os.environ["HF_TOKEN"]; wd=os.environ["ZIP_WORKDIR"]; n=int(os.environ["N_ZIPS"])
zips=discover_zip_names(zip_prefix(sub,"whole"), token=tok)[:n]
r=DownloadZipResolver(zips, token=tok, in_ram=False, workdir=wd)
ds=load_dataset("malaysia-ai/Malaysian-STT", sub, split="train", streaming=True)
print("MEMBERS", len(r.available_members()), "NSHARDS", ds.n_shards, flush=True)
PY
)
  echo "  $info"
  nfiles=$(echo "$info"|grep -oE "NSHARDS [0-9]+"|awk '{print $2}'); nfiles=${nfiles:-1}
  shards=$(( CONC < nfiles ? CONC : nfiles )); [ "$shards" -lt 1 ] && shards=1
  per=$(( (MS_LIMIT + shards - 1)/shards ))
  echo "  -> $shards shards x $per rows"
  for j in $(seq 0 $((shards-1))); do
    launch "msw-$sub-s$j" 14400 KIND=ms CONFIG="$sub" LIMIT="$per" \
      SHARD_IDX=$j SHARD_CNT=$shards ZIP_WORKDIR=$wd N_ZIPS=$N_ZIPS
  done
  wait
  echo "MS_SUBSET_DONE $sub"
  rm -rf "$wd"
done

echo "=== re-split ms last shards ==="
python /root/pod_ms_split.py
echo "MS_REDO_DONE"
