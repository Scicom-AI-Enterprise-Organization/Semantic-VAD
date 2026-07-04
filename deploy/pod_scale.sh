#!/usr/bin/env bash
# Runs ON the pod: full-scale build + push to HF org repo.
set -uo pipefail
export PATH="/root/venv/bin:/root/.local/bin:$PATH"
set -a; [ -f /root/.hf_env ] && . /root/.hf_env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export SVAD_REPO="${SVAD_REPO:-Scicom-intl/semantic-vad-eot}"
export ML_LIMIT="${ML_LIMIT:-1000}"
export MS_LIMIT="${MS_LIMIT:-2000}"
cd /root/semantic-vad
uv pip install --python /root/venv/bin/python -q remotezip >/dev/null 2>&1 || true
python /root/pod_scale.py
