# Semantic-VAD

Build [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data)-compatible
**end-of-turn (EOT) detection** datasets from word-level **forced-alignment** corpora.

Semantic VAD decides *when a user has finished speaking* directly from streaming audio —
not just from "there was silence." A naive silence-VAD cuts the user off at the first
pause; a good EOT model distinguishes a **mid-turn pause** (the speaker is thinking) from
a **true end of turn**. This repo turns forced alignments (words with start/end times)
into a labeled benchmark for exactly that decision.

Background: LiveKit's [Solving end-of-turn detection](https://livekit.com/blog/solving-end-of-turn-detection)
and their [`eot-bench`](https://github.com/livekit/eot-bench) harness.

## The idea

A forced alignment gives you every word's `start`/`end` time. The **gap** between two
consecutive words is silence. The insight the benchmark encodes:

> Within one user turn there can be several pauses ≥ 100 ms (**holds**). Only the pause
> at the very end of the turn is the true **end of turn** (**eot**).

So each output row is one *turn*, carrying an ordered list of `silence_spans`. Per the
eot-bench convention the **last** span is `eot` and every earlier span is `hold` — the
label is positional, not stored, so it's compatible with the eot-bench harness as-is.

```
words:   No, but I would        I am glad for your help today and   this is all I need. Thank you ...
gaps:              └─ 1.0s ─┘                              └─0.2s┘                          ...  └─ trailing ─┘
labels:            HOLD                                    HOLD                                  EOT
```

### Two ways to get the labels

| mode | when to use | how holds/eot are decided |
|------|-------------|---------------------------|
| **`single`** (preferred) | each source row is already one complete utterance/turn (multilingual sentences; Malaysian *streaming* segments) | every internal gap ≥ `min_silence` is a genuine `hold`; the end of the utterance is a genuine `eot`. Label comes from utterance **structure**, not gap size — so it is not trivially recoverable from silence duration (the whole point of semantic VAD). |
| **`segment`** | a row is a long continuous monologue (Malaysian *whole*) | split into pseudo-turns wherever a gap ≥ `turn_gap` (the **sweet spot**). Smaller gaps become `hold`, the boundary gap becomes `eot`. |

> ⚠️ **Caveat for `segment` mode.** Because the turn boundary is *defined by* silence
> duration, labels correlate with gap length there. `single` mode avoids this and yields
> the higher-quality data. LiveKit sidestepped it entirely by using real dialogues where
> turn boundaries are known from who-spoke-when; forced-alignment corpora don't have that,
> so `single` mode (one utterance = one turn) is the closest honest equivalent.

## Output schema (matches `eot-bench-data`)

| column | type | notes |
|--------|------|-------|
| `id` | string | `{lang}__{source_id}__turn_{idx:03d}` |
| `audio` | `Audio(16 kHz)` | mono clip for this turn (zero-padded so the EOT silence is real) |
| `language` | string | ISO code (`en`, `ms`, …) |
| `duration` | float64 | clip length (s) |
| `silence_spans` | `list[{start,end}]` | ordered; **last = `eot`, earlier = `hold`**, each ≥ `min_silence` |
| `words` | `list[{word,start,end}]` | times relative to the clip |
| `messages` | `list[{role,content}]` | conversational context (see limitations) |

## Install

```bash
uv venv --python 3.12 .venv
uv pip install -e .            # or: uv pip install -e ".[dev]" for tests
```

Requires `datasets`, `numpy`, `soundfile` (decodes both WAV and MP3), `huggingface_hub`.
`librosa` is **not** needed — resampling to 16 kHz uses a light numpy linear resampler, and
audio is stored as WAV bytes so **no `torch`/`torchcodec`** is needed to write or read it.
For the Malaysian source also install the `malaysian` extra (`remotezip`):
`uv pip install -e ".[malaysian]"`.

> **Heads-up on where to run.** These corpora are large; build on a machine with good
> bandwidth (we use a RunPod CPU pod — see `deploy/` and `CLAUDE.md`), not a laptop. The
> unit tests are fully offline.

## Usage

### 1. Find the sweet spot (gap threshold)

Sample a corpus and inspect the inter-word gap distribution. The recommended `turn_gap`
is the valley between the "between-word" and "between-turn" modes:

```bash
python -m semantic_vad.analyze --source multilingual --config english --limit 300
# or the console script: svad-analyze --source multilingual --config english
```

Output includes percentiles, a KDE valley estimate, a text histogram, and a
`RECOMMENDED turn_gap = …`. Use it for `--turn-gap` in `segment` mode.

### 2. Build the dataset

```bash
# Multilingual (embedded audio, clean words) — each sentence is one turn:
python -m semantic_vad.build --source multilingual --config english \
    --limit 500 --out data/en.parquet

# Malaysian STT — audio read from the repo's zip archives via HTTP range requests:
python -m semantic_vad.build --source malaysian --config malaysian \
    --malaysian-mode streaming --limit 500 --out data/ms.parquet \
    --malaysian-zips malaysian-segment-0-0.zip --malaysian-max-scan 5000

# Force segment-mode splitting of long recordings at a tuned threshold:
python -m semantic_vad.build --source malaysian --config malaysian \
    --malaysian-mode whole --mode segment --turn-gap 0.7 --out data/ms_whole.parquet
```

Key flags (all have sensible defaults): `--min-silence 0.1`, `--turn-gap 0.7`,
`--eot-trailing 0.5`, `--max-trailing 1.0`, `--lead-in 0.3`, `--min-words 1`,
`--min-hold-spans 0` (raise to keep only "hard" rows with real mid-turn holds),
`--target-sr 16000`. `--mode auto` (default) uses each corpus's suggested mode.

### 3. Use it as a library

```python
from semantic_vad import normalize_words, build_turns, TurnConfig
from semantic_vad.audio import turn_to_row
from semantic_vad.build import write_parquet

words = normalize_words(row["words"])                 # list[Word]
turns = build_turns(words, audio_duration, TurnConfig(mode="single"))
rows = [turn_to_row(t, audio_array, 16000, row_id=f"en__x__turn_{i:03d}", language="en")
        for i, t in enumerate(turns)]
write_parquet(iter(rows), "data/en.parquet")          # eot-bench-compatible parquet
```

> The parquet is written directly with **pyarrow** and stores audio as WAV bytes with the
> HuggingFace `Audio` feature metadata embedded — so `load_dataset(...)` recognizes it as
> an audio dataset, but **no `torch`/`torchcodec`** is needed to write *or* read it
> (`datasets>=5` otherwise imports them to (de)serialize `Audio`). Read audio the eot-bench
> way: `ds.cast_column("audio", Audio(decode=False))` then `soundfile.read(BytesIO(bytes))`.

## Sources

- **[AAdonis/multilingual_audio_alignments](https://huggingface.co/datasets/AAdonis/multilingual_audio_alignments)** —
  13 languages, embedded 16 kHz audio, a clean `words` column `{word,start,end}`. One
  sentence per row → `single` mode. **The recommended starting point.**
- **[malaysia-ai/Malaysian-STT](https://huggingface.co/datasets/malaysia-ai/Malaysian-STT)** —
  whisper-format timestamp strings in `texts` (parsed by `parse_whisper_timestamps`). Audio
  lives **inside ~4.9 GB zip archives** (`malaysian-segment-*.zip`, ~345k members each);
  `semantic_vad.malaysian_audio.ZipAudioResolver` reads individual mp3s over HTTP **range
  requests** (`remotezip`) — no full-archive download. Use `level=word` rows; `streaming`
  mode gives natural per-segment turns, `whole` a long recording. Pass `--malaysian-zips`
  to limit which archives are indexed.

## Running on RunPod + pushing to HF

Large source corpora → build on a CPU pod, not a laptop. `deploy/` has a stdlib-only
control plane (see `CLAUDE.md` for the full flow):

```bash
python3 deploy/runpod_ctl.py create --disk 20   # US CPU pod, SSH via deploy/runpod_key
python3 deploy/runpod_ctl.py wait                # -> READY <ip> <port>
# scp code + deploy/pod_setup.sh, run it (uv → py3.12 venv, deps, offline pytest)
# scp deploy/pod_verify.py + pod_verify.sh, run it (build a few langs + Malaysian, push to HF)
python3 deploy/runpod_ctl.py terminate           # stop billing when done
```

A verification sample built this way (en/es/ja + Malaysian, 280 rows) lives at
**[huseinzolkepliscicom/semantic-vad-eot](https://huggingface.co/datasets/huseinzolkepliscicom/semantic-vad-eot)**.

## Layout

```
semantic_vad/
  schema.py     Word, SilenceSpan, Turn, EOTRow, TurnConfig
  parsers.py    parse_whisper_timestamps (Malaysian), normalize_words (words-list)
  turns.py      compute_gaps, build_turns  ← the hold/eot labeling logic
  analyze.py    analyze_gaps + `-m semantic_vad.analyze` CLI (sweet spot)
  audio.py      resample_linear, slice_window (zero-pads), turn_to_row, encode_wav
  sources.py    iter_multilingual / iter_malaysian streaming adapters
  malaysian_audio.py  ZipAudioResolver — read mp3s from remote zips via range requests
  build.py      build_rows + `-m semantic_vad.build` CLI (writes parquet with pyarrow)
tests/          parser/turn/analyze/build unit tests on real captured fixtures (offline)
deploy/         RunPod control plane + pod setup/build/verify scripts
```

## Limitations & honest notes

- **`messages` context.** These corpora are monologue/utterance recordings without speaker
  diarization, so `messages` defaults to a single `{"role":"user","content": <transcript>}`.
  The eot-bench harness treats `messages` as optional context; populate real dialogue
  history if you have it.
- **`segment`-mode label leakage** — see the caveat above; prefer `single` mode.
- **Trailing silence.** If a source clip ends right after the last word, the EOT span is
  created by zero-padding the audio to `eot_trailing` seconds. Real recorded trailing
  silence is used when present (clamped to `max_trailing`).
- **Resampling** is linear interpolation (fine for timing/coarse-spectral EOT cues), not
  hi-fi — swap in `soxr`/`librosa` if you need studio-quality resampling.

## Testing

```bash
.venv/bin/python -m pytest -q
```
