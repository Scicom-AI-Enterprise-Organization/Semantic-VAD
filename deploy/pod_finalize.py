"""Write the dataset card after all shards are uploaded. Runs ON the pod."""
import os
import sys

from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = os.environ.get("SVAD_REPO", "Scicom-intl/semantic-vad-eot")
api = HfApi(token=TOKEN)

files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
         if f.startswith("data/") and f.endswith(".parquet")]
ml = sorted({f.split("/")[1].split("-")[1] for f in files if f.startswith("data/ml-")})
ms = sorted({f.split("/")[1].split("-")[1] for f in files if f.startswith("data/ms-")})
ML_ISO = {"english": "en", "french": "fr", "german": "de", "italian": "it", "japanese": "ja",
          "korean": "ko", "mandarin": "zh", "polish": "pl", "portuguese": "pt", "russian": "ru",
          "spanish": "es", "thai": "th", "turkish": "tr"}
langs = sorted({ML_ISO.get(l, l) for l in ml} | ({"ms"} if ms else set()))
lang_yaml = "\n".join(f"- {l}" for l in langs)

card = f"""---
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
    path: data/*.parquet
---

# Semantic-VAD EOT

End-of-turn (semantic VAD) turns built from word-level forced alignments, schema-compatible
with [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data).

Each row is one user **turn**: an audio clip (16 kHz **mp3**), its `words`, and ordered
`silence_spans`. Per the eot-bench convention the **last** silence span is the true
end-of-turn (`eot`); every earlier span is a mid-turn `hold` pause. Labels are positional
(not stored). Train per decision point: crop audio at each span, label the last `eot`=1 and
the rest `hold`=0.

**Sources:** `AAdonis/multilingual_audio_alignments` (langs: {', '.join(sorted(ml))}) and
`malaysia-ai/Malaysian-STT` (word-level, subsets: {', '.join(sorted(ms)) or 'n/a'}; no synthetic).

```python
from datasets import load_dataset, Audio
import soundfile as sf, io
ds = load_dataset("{REPO}", split="validation").cast_column("audio", Audio(decode=False))
arr, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]))
```
"""
with open("/root/data/README.md", "w") as f:
    f.write(card)
api.upload_file(path_or_fileobj="/root/data/README.md", repo_id=REPO, repo_type="dataset",
                path_in_repo="README.md")
print(f"finalized: {len(files)} shards, langs={langs}", flush=True)
sys.stdout.flush()
os._exit(0)
