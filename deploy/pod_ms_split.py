"""Split each Malaysian type's last shard (data/ms-<sub>-*) into 50/25/25 train/val/test.

Leaves all-but-last shards in data/ (train base); writes the split of the last shard into
data/train, data/validation, data/test; deletes the original last shard from data/.
Mirrors the multilingual split so the existing dataset-card globs keep working.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

REPO = "Scicom-intl/semantic-vad-eot"
TOK = os.environ["HF_TOKEN"]
api = HfApi(token=TOK)

files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
         if f.startswith("data/ms-") and f.endswith(".parquet")]
g = defaultdict(list)
for f in files:
    g["-".join(f.split("/")[-1].split("-")[:2])].append(f)

for k, fs in g.items():
    last = sorted(fs)[-1]
    name = last.split("/")[-1]
    local = hf_hub_download(REPO, last, repo_type="dataset", token=TOK)
    t = pq.read_table(local)
    n = t.num_rows
    t = t.take(np.random.default_rng(42).permutation(n))
    a, b = n // 2, n // 2 + n // 4
    for split, tab in {"train": t.slice(0, a), "validation": t.slice(a, b - a),
                       "test": t.slice(b, n - b)}.items():
        out = f"/root/{split}-{name}"
        pq.write_table(tab, out)
        api.upload_file(path_or_fileobj=out, path_in_repo=f"data/{split}/{name}",
                        repo_id=REPO, repo_type="dataset")
        os.remove(out)
    os.remove(local)
    api.create_commit(REPO, repo_type="dataset",
                      operations=[CommitOperationDelete(path_in_repo=last)],
                      commit_message=f"split {k} last shard 50/25/25")
    print(f"  split {k}: n={n} -> train {a}, val {b-a}, test {n-b}", flush=True)
print("MS_SPLIT_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
