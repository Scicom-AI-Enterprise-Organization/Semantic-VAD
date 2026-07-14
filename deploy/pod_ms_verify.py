"""Verify Malaysian whole-mode + sentence turns: real audio + complete-sentence turns."""
import os
import sys

import numpy as np

sys.path.insert(0, "/root/semantic-vad")
from semantic_vad.build import build_rows  # noqa: E402
from semantic_vad.schema import TurnConfig  # noqa: E402

cfg = TurnConfig(mode="sentence", turn_gap=1.5, min_silence=0.1)
rows = list(build_rows("malaysian", os.environ.get("SUB", "dialects"), cfg, mode="sentence",
                       limit=12, streaming=True, hf_token=os.environ["HF_TOKEN"],
                       malaysian_mode="whole", malaysian_backend="download",
                       malaysian_n_zips=2, malaysian_max_scan=50000))
print(f"built {len(rows)} turns", flush=True)
sr = 16000
tails = []
for r in rows:
    a = np.asarray(r.audio)
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    tail = a[-int(0.05 * sr):] if a.size else a          # last 50ms
    tail_rms = float(np.sqrt(np.mean(tail ** 2))) if tail.size else 0.0
    tails.append(tail_rms)
    wl = r.words[-1]["end"] if r.words else 0.0
    print("  dur=%.2f rms=%.3f TAIL_rms=%.4f holds=%d | %r"
          % (r.duration, rms, tail_rms, len(r.silence_spans) - 1,
             r.messages[0]["content"][:56]), flush=True)
tails = np.asarray(tails)
print("TAIL RMS summary: mean=%.4f p90=%.4f max=%.4f  (low == ends in silence)"
      % (float(tails.mean()), float(np.percentile(tails, 90)), float(tails.max())), flush=True)
sys.stdout.flush()
os._exit(0)
