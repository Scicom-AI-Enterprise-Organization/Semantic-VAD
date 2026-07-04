"""Full-scale build: all 13 multilingual languages + Malaysian, pushed to HF as
per-language parquet shards (one `validation` split, `language` column). Runs ON the pod.

Uploads each shard then deletes it locally so the 20 GB container disk stays bounded.
Target repo via SVAD_REPO env (default the org dataset).
"""

import gc
import io
import json
import os
import sys
from collections import Counter

import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, "/root/semantic-vad")
from semantic_vad.build import build_rows, write_parquet  # noqa: E402
from semantic_vad.schema import TurnConfig  # noqa: E402

DATA = "/root/data"
os.makedirs(DATA, exist_ok=True)
TOKEN = os.environ["HF_TOKEN"]

ML_LANGS = ["english", "french", "german", "italian", "japanese", "korean", "mandarin",
            "polish", "portuguese", "russian", "spanish", "thai", "turkish"]
ML_LIMIT = int(os.environ.get("ML_LIMIT", "1000"))
MS_LIMIT = int(os.environ.get("MS_LIMIT", "2000"))
MS_ZIPS = [f"malaysian-segment-{i}-0.zip" for i in range(4)]  # index 4 archives for coverage


def card(counts):
    langs = sorted(counts)
    lang_yaml = "\n".join(f"- {l}" for l in langs)
    rows = "\n".join(f"| {l} | {counts[l]} |" for l in langs)
    total = sum(counts.values())
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
pretty_name: Semantic-VAD EOT
configs:
- config_name: default
  data_files:
  - split: validation
    path: data/validation-*.parquet
---

# Semantic-VAD EOT

{total} user turns across {len(langs)} languages, schema-compatible with
[`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data).

Each row is one user **turn** built from word-level forced alignments: an audio clip (16 kHz),
its `words`, and ordered `silence_spans`. Per the eot-bench convention the **last** silence
span is the true end-of-turn (`eot`) and every earlier span is a mid-turn `hold` pause.

Built from `AAdonis/multilingual_audio_alignments` and `malaysia-ai/Malaysian-STT`.

| language | rows |
|----------|------|
{rows}
| **total** | **{total}** |

```python
from datasets import load_dataset, Audio
import soundfile as sf, io
ds = load_dataset("Scicom-intl/semantic-vad-eot", split="validation").cast_column("audio", Audio(decode=False))
arr, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
```
"""


def main():
    from huggingface_hub import HfApi, whoami

    who = whoami(token=TOKEN)
    repo = os.environ.get("SVAD_REPO", "Scicom-intl/semantic-vad-eot")
    api = HfApi(token=TOKEN)
    print(f"[hf] user={who.get('name')} -> repo={repo}", flush=True)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)

    # Clean any existing data/ shards so old samples don't linger in the split.
    try:
        for f in api.list_repo_files(repo, repo_type="dataset"):
            if f.startswith("data/") and f.endswith(".parquet"):
                api.delete_file(f, repo, repo_type="dataset")
                print(f"[clean] removed {f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[clean] skip: {e}", flush=True)

    plan = [("multilingual", l, dict(limit=ML_LIMIT)) for l in ML_LANGS]
    plan.append(("malaysian", "malaysian",
                 dict(limit=MS_LIMIT, malaysian_mode="streaming",
                      malaysian_zips=MS_ZIPS, malaysian_max_scan=200000)))
    total_shards = len(plan)

    counts: dict[str, int] = {}
    idx = 0
    for source, config, kw in plan:
        out = f"{DATA}/shard.parquet"
        try:
            rows = build_rows(source, config, TurnConfig(mode="single"),
                              mode="auto", streaming=True, hf_token=TOKEN, **kw)
            n = write_parquet(rows, out)
        except SystemExit as e:
            print(f"[skip] {config}: {e}", flush=True)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[error] {config}: {e!r}", flush=True)
            continue

        c = Counter(pq.read_table(out, columns=["language"]).column("language").to_pylist())
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        dst = f"data/validation-{idx:05d}-of-{total_shards:05d}.parquet"
        api.upload_file(path_or_fileobj=out, repo_id=repo, repo_type="dataset", path_in_repo=dst)
        print(f"[done] {config}: {n} rows -> {dst} | running total={sum(counts.values())}",
              flush=True)
        os.remove(out)
        idx += 1
        gc.collect()

    with open(f"{DATA}/README.md", "w") as f:
        f.write(card(counts))
    api.upload_file(path_or_fileobj=f"{DATA}/README.md", repo_id=repo, repo_type="dataset",
                    path_in_repo="README.md")
    print(f"SCALE_DONE {json.dumps(counts)}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
