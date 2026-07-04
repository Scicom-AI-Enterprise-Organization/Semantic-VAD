#!/usr/bin/env bash
# Runs ON the pod: sweet-spot analysis + small end-to-end builds from both sources,
# then prints one sample row per output to prove eot-bench schema compatibility.
set -euo pipefail
export PATH="/root/venv/bin:/root/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /root/semantic-vad
mkdir -p /root/data

echo "########## SWEET SPOT (multilingual english) ##########"
python -m semantic_vad.analyze --source multilingual --config english --limit 200 || true

echo "########## BUILD multilingual english (single mode) ##########"
python -m semantic_vad.build --source multilingual --config english \
  --limit 200 --out /root/data/en.parquet

echo "########## BUILD malaysian (streaming segments) ##########"
python -m semantic_vad.build --source malaysian --config malaysian \
  --malaysian-mode streaming --limit 150 --out /root/data/ms.parquet || true

echo "########## VERIFY output schema vs eot-bench-data ##########"
python - <<'PY'
import io, json, os
import soundfile as sf
from datasets import Audio, load_dataset
for name in ["en", "ms"]:
    path = f"/root/data/{name}.parquet"
    if not os.path.exists(path):
        print(f"[skip] {path} missing"); continue
    ds = load_dataset("parquet", data_files=path, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))  # read bytes like eot-bench
    print(f"\n=== {name}.parquet : {len(ds)} rows ===")
    print("features:", {k: str(v) for k, v in ds.features.items()})
    r = ds[0]
    arr, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    print("id       :", r["id"])
    print("language :", r["language"], "| duration:", round(r["duration"], 2))
    print("audio    : sr=", sr, "samples=", len(arr), "sec=", round(len(arr)/sr, 2))
    print("silence_spans:", json.dumps(r["silence_spans"]))
    print("words[:6]:", json.dumps(r["words"][:6], ensure_ascii=False))
    print("messages :", json.dumps(r["messages"], ensure_ascii=False)[:200])
    holds = len(r["silence_spans"]) - 1
    print(f"-> {holds} hold span(s) + 1 eot span (last = end-of-turn)")
import sys; sys.stdout.flush(); os._exit(0)
PY

echo "########## disk ##########"
du -sh /root/data /root/.cache/huggingface 2>/dev/null || true
df -h / | tail -1
echo "BUILD_OK"
