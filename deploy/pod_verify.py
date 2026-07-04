"""Small end-to-end verification build + push to HF Hub. Runs ON the pod.

Builds a few languages from the multilingual corpus and a Malaysian sample (audio read
from one zip archive), concatenates into a single `validation` split with a `language`
column (like eot-bench-data's `all` config), and uploads to <hf-user>/semantic-vad-eot.
"""

import io
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, "/root/semantic-vad")
from semantic_vad.build import build_rows, write_parquet  # noqa: E402
from semantic_vad.schema import TurnConfig  # noqa: E402

DATA = "/root/data"
os.makedirs(DATA, exist_ok=True)
TOKEN = os.environ.get("HF_TOKEN")
REPO_NAME = os.environ.get("SVAD_REPO")  # optional override


def build(source, config, out, **kw):
    cfg = TurnConfig(mode="single")
    rows = build_rows(source, config, cfg, mode="auto", streaming=True, hf_token=TOKEN, **kw)
    n = write_parquet(rows, out)
    print(f"[build] {source}:{config} -> {n} rows -> {out}", flush=True)
    return n


def main():
    plan = [
        ("multilingual", "english", f"{DATA}/en.parquet", dict(limit=80)),
        ("multilingual", "spanish", f"{DATA}/es.parquet", dict(limit=80)),
        ("multilingual", "japanese", f"{DATA}/ja.parquet", dict(limit=80)),
        ("malaysian", "malaysian", f"{DATA}/ms.parquet",
         dict(limit=40, malaysian_mode="streaming",
              malaysian_zips=["malaysian-segment-0-0.zip"], malaysian_max_scan=800)),
    ]
    built = []
    for source, config, out, kw in plan:
        try:
            if build(source, config, out, **kw):
                built.append(out)
        except SystemExit as e:
            print(f"[skip] {config}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {config}: {e!r}", flush=True)

    # Concatenate into one validation split (schema + HF metadata identical across files).
    tables = [pq.read_table(p) for p in built]
    combined = pa.concat_tables(tables)
    combined = combined.replace_schema_metadata(tables[0].schema.metadata)
    all_path = f"{DATA}/validation-00000-of-00001.parquet"
    pq.write_table(combined, all_path)

    from collections import Counter
    langs = Counter(combined.column("language").to_pylist())
    print(f"[combined] {combined.num_rows} rows | languages: {dict(langs)}", flush=True)

    # Sanity: first row's audio decodes.
    r0 = combined.slice(0, 1).to_pylist()[0]
    arr, sr = sf.read(io.BytesIO(r0["audio"]["bytes"]), dtype="float32")
    print(f"[sanity] {r0['id']} dur={r0['duration']} sr={sr} samples={len(arr)} "
          f"spans={json.dumps(r0['silence_spans'])}", flush=True)

    # Push to HF Hub.
    from huggingface_hub import HfApi, whoami

    user = whoami(token=TOKEN)["name"]
    repo = REPO_NAME or f"{user}/semantic-vad-eot"
    api = HfApi(token=TOKEN)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)

    readme = _dataset_card(sorted(langs), combined.num_rows)
    with open(f"{DATA}/README.md", "w") as f:
        f.write(readme)

    api.upload_file(path_or_fileobj=all_path, repo_id=repo, repo_type="dataset",
                    path_in_repo="data/validation-00000-of-00001.parquet")
    api.upload_file(path_or_fileobj=f"{DATA}/README.md", repo_id=repo, repo_type="dataset",
                    path_in_repo="README.md")
    print(f"PUSHED https://huggingface.co/datasets/{repo}", flush=True)


def _dataset_card(langs, n_rows):
    lang_yaml = "\n".join(f"- {l}" for l in langs)
    return f"""---
license: cc-by-4.0
task_categories:
- voice-activity-detection
language:
{lang_yaml}
tags:
- end-of-turn
- semantic-vad
- eot
- turn-detection
pretty_name: Semantic-VAD EOT (verification sample)
configs:
- config_name: default
  data_files:
  - split: validation
    path: data/validation-*.parquet
---

# Semantic-VAD EOT — verification sample

{n_rows} rows, schema-compatible with [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data).
Each row is one user **turn** built from word-level forced alignments: an audio clip, its
`words`, and ordered `silence_spans` where the **last** span is the true end-of-turn (`eot`)
and earlier spans are mid-turn `hold` pauses.

Built with [Semantic-VAD](https://github.com/huseinzol05) from
`AAdonis/multilingual_audio_alignments` and `malaysia-ai/Malaysian-STT`. This is a small
sample to verify the pipeline; see the tool repo for full-scale builds.

Read audio the eot-bench way:
```python
from datasets import load_dataset, Audio
import soundfile as sf, io
ds = load_dataset("REPO", split="validation").cast_column("audio", Audio(decode=False))
arr, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
```
"""


if __name__ == "__main__":
    main()
    print("VERIFY_DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)  # streaming readers spawn native threads that crash at finalization
