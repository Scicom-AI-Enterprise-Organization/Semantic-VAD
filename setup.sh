uv venv --python 3.12 --allow-existing
uv pip install huggingface_hub
uv pip install -r requirements.txt
uv pip install git+https://github.com/livekit/eot-bench.git # for eot-harness benchmarking