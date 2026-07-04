"""Measure mp3 throughput (multilingual streaming + Malaysian download backend). On the pod."""
import os, sys, time
sys.path.insert(0, "/root/semantic-vad")
from semantic_vad.build import build_rows, write_parquet
from semantic_vad.schema import TurnConfig
TOK = os.environ["HF_TOKEN"]

t = time.time()
n = write_parquet(build_rows("multilingual", "english", TurnConfig(mode="single"),
                             mode="auto", limit=3000, streaming=True, hf_token=TOK),
                  "/root/data/b_en.parquet", audio_format="mp3")
dt = time.time() - t
print(f"ML(mp3) english: {n} rows in {dt:.1f}s = {n/dt:.1f} rows/s", flush=True)

# Malaysian download backend: measure zip download + processing separately.
from semantic_vad.malaysian_audio import discover_zip_names, DownloadZipResolver
zips = discover_zip_names("malaysian-segment", token=TOK)[:3]
t = time.time()
res = DownloadZipResolver(zips, token=TOK, in_ram=False)
print(f"MS zip download+open ({len(zips)} zips): {time.time()-t:.1f}s, members={len(res.available_members())}", flush=True)
res.close()

t = time.time()
n = write_parquet(build_rows("malaysian", "malaysian", TurnConfig(mode="single"),
                             mode="auto", limit=1000, streaming=True, hf_token=TOK,
                             malaysian_mode="streaming", malaysian_backend="download",
                             malaysian_n_zips=3, malaysian_max_scan=50_000_000),
                  "/root/data/b_ms.parquet", audio_format="mp3")
dt = time.time() - t
print(f"MS(mp3, download) malaysian: {n} rows in {dt:.1f}s = {n/dt:.1f} rows/s (incl 3-zip dl)", flush=True)
sys.stdout.flush(); os._exit(0)
