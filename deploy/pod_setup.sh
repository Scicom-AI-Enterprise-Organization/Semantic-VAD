#!/usr/bin/env bash
# Runs ON the RunPod CPU pod. Provisions Python 3.12 via uv, installs the package,
# runs the offline unit tests. Everything lives under / (container disk), no /workspace.
set -euo pipefail

cd /root
echo "== extract code =="
rm -rf /root/semantic-vad && mkdir -p /root/semantic-vad
tar xzf /root/svad_code.tgz -C /root/semantic-vad --no-same-owner --warning=no-unknown-keyword
cd /root/semantic-vad

echo "== install uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

echo "== create py3.12 venv =="
uv venv --python 3.12 /root/venv
export PATH="/root/venv/bin:$PATH"

echo "== install deps (soundfile decodes wav+mp3; no librosa) =="
uv pip install --python /root/venv/bin/python \
  datasets pyarrow numpy soundfile huggingface_hub pytest remotezip lameenc hf_xet

echo "== unit tests (offline) =="
/root/venv/bin/python -m pytest -q

echo "SETUP_OK"
