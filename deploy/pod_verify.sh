#!/usr/bin/env bash
# Runs ON the pod: sources HF token, ensures remotezip, runs the verify build + push.
set -uo pipefail
export PATH="/root/venv/bin:/root/.local/bin:$PATH"
set -a; [ -f /root/.hf_env ] && . /root/.hf_env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
cd /root/semantic-vad
uv pip install --python /root/venv/bin/python -q remotezip >/dev/null 2>&1 || true
python /root/pod_verify.py
