"""Split each type's last shard (in data/valid/) into 50% train / 25% validation / 25% test.

Runs ON a pod. For each file in data/valid/: download (Xet), deterministically shuffle rows,
slice 50/25/25, upload to data/train|validation|test/<name>, delete data/valid/ afterward.
Parquet schema (incl. HF `huggingface` feature metadata) is preserved by pyarrow.
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

REPO = "Scicom-intl/semantic-vad-eot"
TOK = os.environ["HF_TOKEN"]
api = HfApi(token=TOK)

valid = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset")
               if f.startswith("data/valid/") and f.endswith(".parquet"))
print(f"splitting {len(valid)} last-shards 50/25/25", flush=True)

for f in valid:
    name = f.split("/")[-1]
    local = hf_hub_download(REPO, f, repo_type="dataset", token=TOK)
    t = pq.read_table(local)
    n = t.num_rows
    perm = np.random.default_rng(42).permutation(n)
    t = t.take(perm)  # deterministic shuffle so val/test aren't a contiguous tail
    a, b = n // 2, n // 2 + n // 4
    parts = {"train": t.slice(0, a), "validation": t.slice(a, b - a), "test": t.slice(b, n - b)}
    for split, tab in parts.items():
        out = f"/root/{split}-{name}"
        pq.write_table(tab, out)
        api.upload_file(path_or_fileobj=out, path_in_repo=f"data/{split}/{name}",
                        repo_id=REPO, repo_type="dataset")
        os.remove(out)
    os.remove(local)
    print(f"  {name}: n={n} -> train {a}, val {b-a}, test {n-b}", flush=True)

api.create_commit(REPO, repo_type="dataset",
                  operations=[CommitOperationDelete(path_in_repo=f) for f in valid],
                  commit_message="Replace data/valid last-shards with 50/25/25 train/validation/test")
print("SPLIT_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
