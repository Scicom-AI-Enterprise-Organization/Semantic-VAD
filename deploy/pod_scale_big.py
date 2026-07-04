"""One scale worker (env-driven). Runs ON the pod; a launcher spawns many in parallel.

Builds one unit of work -- a whole multilingual language, or one shard of a Malaysian
subset -- writing fixed-size mp3 parquet shards, uploading each to HF, deleting locally
(so the 20 GB disk never fills).

Env:
  HF_TOKEN     (required)
  SVAD_REPO    target repo (default Scicom-intl/semantic-vad-eot)
  KIND         "ml" (multilingual) | "ms" (Malaysian)
  CONFIG       language (ml) or subset (ms)
  LIMIT        max rows this worker emits
  AUDIO_FORMAT default mp3
  SHARD_ROWS   rows per uploaded shard (default 20000)
  KEY          filename key for uploaded shards (default {KIND}-{CONFIG})
  # Malaysian only:
  N_ZIPS       archives to use (default 3)
  ZIP_WORKDIR  dir holding pre-downloaded zips (shared across shards)
  SHARD_IDX / SHARD_CNT   partition the row stream across parallel shard workers
"""

import gc
import itertools
import os
import sys
import time

sys.path.insert(0, "/root/semantic-vad")
from semantic_vad.build import build_rows, write_parquet  # noqa: E402
from semantic_vad.malaysian_audio import discover_zip_names  # noqa: E402
from semantic_vad.schema import TurnConfig  # noqa: E402

TOKEN = os.environ["HF_TOKEN"]
REPO = os.environ.get("SVAD_REPO", "Scicom-intl/semantic-vad-eot")
KIND = os.environ["KIND"]
CONFIG = os.environ["CONFIG"]
LIMIT = int(os.environ["LIMIT"])
AUDIO_FORMAT = os.environ.get("AUDIO_FORMAT", "mp3")
SHARD_ROWS = int(os.environ.get("SHARD_ROWS", "20000"))
KEY = os.environ.get("KEY", f"{KIND}-{CONFIG}")
DATA = "/root/data"
os.makedirs(DATA, exist_ok=True)


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def make_rows():
    cfg = TurnConfig(mode="single")
    if KIND == "ml":
        return build_rows("multilingual", CONFIG, cfg, mode="auto",
                          limit=LIMIT, streaming=True, hf_token=TOKEN)
    n_zips = int(os.environ.get("N_ZIPS", "3"))
    workdir = os.environ.get("ZIP_WORKDIR", f"{DATA}/zips-{CONFIG}")
    zips = discover_zip_names(f"{CONFIG}-segment", token=TOKEN)[:n_zips]
    shard = None
    if os.environ.get("SHARD_CNT"):
        shard = (int(os.environ["SHARD_IDX"]), int(os.environ["SHARD_CNT"]))
    return build_rows("malaysian", CONFIG, cfg, mode="auto",
                      limit=LIMIT, streaming=True, hf_token=TOKEN,
                      malaysian_mode="streaming", malaysian_zips=zips,
                      malaysian_backend="download", malaysian_n_zips=n_zips,
                      malaysian_shard=shard, malaysian_max_scan=50_000_000)


def main():
    from huggingface_hub import HfApi

    api = HfApi(token=TOKEN)
    suffix = f"-s{os.environ.get('SHARD_IDX')}" if os.environ.get("SHARD_CNT") else ""
    t0 = time.time()
    total = 0
    for si, chunk in enumerate(chunked(make_rows(), SHARD_ROWS)):
        shard_path = f"{DATA}/{KEY}{suffix}-{si:05d}.parquet"
        n = write_parquet(iter(chunk), shard_path, audio_format=AUDIO_FORMAT)
        dst = f"data/{KEY}{suffix}-{si:05d}.parquet"
        api.upload_file(path_or_fileobj=shard_path, repo_id=REPO, repo_type="dataset",
                        path_in_repo=dst)
        os.remove(shard_path)
        total += n
        rate = total / max(1e-9, time.time() - t0)
        print(f"[{KEY}{suffix}] shard {si} +{n} -> {dst} | total={total} ({rate:.1f} rows/s)",
              flush=True)
        del chunk
        gc.collect()
    print(f"[{KEY}{suffix}] DONE total={total} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
