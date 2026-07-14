# CLAUDE.md — Semantic-VAD

Guide for future Claude Code sessions in this repo.

## What this is

Tooling to build [`livekit/eot-bench-data`](https://huggingface.co/datasets/livekit/eot-bench-data)-compatible
**end-of-turn (EOT) / semantic-VAD** datasets from word-level **forced-alignment** corpora.
Each output row is one user *turn*: an audio clip + its `words` + ordered `silence_spans`
where the **last** span is the true end-of-turn (`eot`) and earlier spans are mid-turn
`hold` pauses. See `README.md` for methodology.

## Architecture (`semantic_vad/`)

- `schema.py` — `Word`, `SilenceSpan`, `Turn`, `EOTRow`, `TurnConfig` (all tunables).
- `parsers.py` — `parse_whisper_timestamps` (Malaysian `<|t|>word<|t|>` format),
  `normalize_words` (AAdonis `{word,start,end}` list). Fully unit-tested, no network.
- `turns.py` — **the core logic.** `compute_gaps`, `build_turns`. Three modes:
  `single` (whole utterance = one turn; internal gaps→hold, end→eot), `segment` (split at
  gaps ≥ `turn_gap`), and `sentence` (split at sentence-final `. ? !` with an abbreviation
  guard + a `≥ turn_gap` fallback — **used for Malaysian**; continuous speech has no bimodal
  gap valley so a gap threshold cuts mid-sentence).
- `analyze.py` — `analyze_gaps` + `python -m semantic_vad.analyze` CLI. Finds the
  "sweet spot" gap threshold via a numpy-only KDE valley over log-gaps. NOTE: returns ~0.4 s
  for conversational corpora (no valley) → too small; that's why Malaysian uses `sentence`.
- `audio.py` — `resample_linear` (16 kHz, no librosa), `slice_window` (zero-pads), `encode_wav`/
  `encode_audio` (mp3 via lameenc), **`speech_end_index`** (energy VAD — finds where speech
  ends), and `turn_to_row` which: VAD-trims the tail to end ~0.15 s into silence, caps
  trailing/lead-in at the real gap (`eot_guard`, never reaches the neighboring word), and
  **returns `None` to DROP a turn whose tail is still loud** (music / no-pause = no real `eot`
  silence). `build_rows` skips the `None`s. A row must end in silence.
- `sources.py` — streaming adapters `iter_multilingual`, `iter_malaysian` + `SOURCES`.
- `malaysian_audio.py` — audio lives in multi-GB zips. `DownloadZipResolver` (production):
  download whole-recording zips (Xet) and read mp3s locally. `ZipAudioResolver`: per-mp3 HTTP
  range requests (`remotezip`, `malaysian` extra). `zip_prefix()` maps config→archive prefix
  (`parliament`→`parlimen`, `science_english`→`science`); members are scattered across zips.
- `build.py` — `build_rows` + `python -m semantic_vad.build` CLI → writes parquet **directly
  with pyarrow**, audio as WAV bytes + embedded HF `Audio(16kHz)` feature metadata. Do NOT
  use `datasets.Dataset.from_list`/`push_to_hub` for audio — datasets>=5 imports
  torch/torchcodec to (de)serialize `Audio`. `main()` ends with `os._exit(0)` because
  streaming readers/libsndfile crash at interpreter finalization (harmless, after the write).

Data flow: `source row → normalize/parse → build_turns → turn_to_row → EOTRow → parquet`.

## Training (`semantic_vad/training/`)

Fine-tunes the model this dataset feeds: **Whisper encoder → linear adapter → Qwen3**,
end-of-turn read from a `<|eot|>`/`<|hold|>` marker-token logprob (README's "the branch we
build"; serving contract in `INTEGRATION.md`). Recipe cribbed from malaysia-ai/malaya
`session/audiollm` (audio-token injection) but with three deliberate differences the user
asked for: **no LoRA** (full fine-tune), **flash-attention varlen** packing (`cu_seq_lens_q/k`
+ `max_length_q/k`, à la Multilingual-TTS `qwen3_adamw.py`) — **not** the SDPA
`block_diagonal_concat_inverted` 4D mask — and an **optional** frozen Whisper encoder.

- `prompt.py` / `packing.py` — **pure Python, torch-free** (so `import semantic_vad` stays
  offline). `build_example` lays out the chat: system + user(`<|audio_bos|>`+`<|AUDIO|>`*N+
  `<|audio_eos|>`) + assistant(**marker first**, then transcript). Only the assistant span is
  supervised; marker-first = single-probe serving read. `num_audio_tokens((mel−1)//2+1)` must
  equal the encoder's per-clip valid-frame count. `build_cu_seqlens` = varlen boundaries.
- `modeling.py` — `WhisperQwen3(Qwen3ForCausalLM)` + `self.encoder` (HF `WhisperEncoder`) +
  `self.projection`. `forward` scatters projected audio frames into `inputs_embeds` at
  `<|AUDIO|>` positions, runs Qwen3 with `attention_mask=None` and the varlen kwargs passed
  through `**kwargs`, and takes a fused-linear-CE (Liger) loss normalized by
  `num_items_in_batch`. Global shift is safe across packed boundaries (examples start on
  masked prompt tokens). `freeze_encoder()` optional.
- `data.py` — `SemanticVADDataset` reads the eot parquet (`Audio(decode=False)` + soundfile,
  **no torchcodec**), expands each turn into one `<|eot|>` example (full clip) **plus one
  `<|hold|>` per internal hold span** (audio truncated at the pause, words up to it), mel-
  featurizes + tokenizes on the fly. `collate_packed` multipacks a micro-batch into one
  `[1, total]` varlen sequence. `per_device_train_batch_size` = examples packed per sequence.
- `train.py` — `python -m semantic_vad.training.train` (also `svad-train`). HF `Trainer`,
  AdamW, loads Qwen3 + injects pretrained Whisper encoder, adds special tokens, resizes
  embeddings. Runs on a **GPU pod** — deps in the `[train]` extra (torch, transformers>=4.51,
  accelerate, liger-kernel) + `flash-attn` installed separately (`--no-build-isolation`,
  needs nvcc). Launch: `deploy/train_qwen3.sh` (`torchrun`, env-tunable QWEN3_NAME/
  WHISPER_NAME/TRAIN_FILES/FREEZE_ENCODER). Never on the laptop.

## Sources

- **AAdonis/multilingual_audio_alignments** — 13 langs, embedded 16 kHz audio, clean
  `words`. One sentence/row → `single` mode. **Primary, fully validated path.**
- **malaysia-ai/Malaysian-STT** — whisper-format `texts`, `level=word`, **no synthetic subset**.
  Build with **`whole` mode + `sentence` turns** (production): the `streaming` per-segment mp3s
  are ASR chunks that end mid-sentence (bad `eot`), so use the whole-recording mp3 + continuous
  word timeline and split at punctuation. Audio downloaded via `DownloadZipResolver`.

## RUNNING — use RunPod, NOT the laptop

The user's laptop has **bad internet and low storage**: do NOT download datasets or build
locally. Unit tests (`pytest`) are offline and safe to run anywhere. All dataset building
runs on a **RunPod CPU pod, US region**.

- Secrets live in `.env` (gitignored): `RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`.
  Never print their values.
- **Always work under `/`** on the pod (e.g. `/root/semantic-vad`, `/root/data`), never
  `/workspace`. Pods are created with `volumeInGb: 0` (no network volume) — CPU container
  disk caps at **20 GB**.
- Pod Python is 3.8; we provision **3.12 via `uv`** into `/root/venv`.

### Deploy workflow (`deploy/`)

```bash
python3 deploy/runpod_ctl.py create --disk 20   # launch US CPU pod (saves deploy/pod.json)
python3 deploy/runpod_ctl.py wait                # -> prints "READY <ip> <port>"
# scp deploy/pod_setup.sh + code tarball, then: nohup bash pod_setup.sh (uv venv, deps, pytest)
# scp deploy/pod_verify.py + pod_verify.sh, then: nohup bash pod_verify.sh
#   (builds a few langs + Malaysian, concatenates to one validation split, pushes to HF)
python3 deploy/runpod_ctl.py terminate           # tear down when done (stops billing!)
```

- `deploy/runpod_ctl.py` — stdlib-only RunPod REST client: `create/wait/status/ssh-info/
  terminate`. Reads `.env`. Uses `deploy/runpod_key{,.pub}` (ed25519, gitignored) injected
  via the pod's `PUBLIC_KEY` env for SSH. CPU container disk is capped at 20 GB.
- **HF token on the pod**: RunPod env vars are NOT visible in SSH sessions, so scp the
  `HF_TOKEN=` line from `.env` to `/root/.hf_env` (chmod 600) and `source` it in pod scripts.
- `deploy/pod_verify.py` — small build + concat + push (verification sample).
- **Full-scale run** (`deploy/pod_launch.sh`): Phase 1 multilingual (one parallel worker per
  language via `pod_scale_big.py`), Phase 2 Malaysian (per subset: predownload a few zips to a
  shared dir, then `CONC` sharded workers). Uploads mp3 shards to `Scicom-intl/semantic-vad-eot`
  and deletes locally. `ML_LIMIT`/`MS_LIMIT`/`CONC`/`N_ZIPS`/`SHARD_ROWS` env-tunable;
  `pod_finalize.py` writes the card. For >20 GB disk use `--flavor cpu3c` (allows 60 GB) — cpu3g
  caps at 20; cpu5c was often unavailable.
- **Audio = mp3** (`lameenc`, 32 kbps, quality 7 ≈ 70 rows/s/core; quality 2 was ~40). Format via
  `write_parquet(audio_format=...)` / `--audio-format`; WAV/FLAC go through soundfile.
- **Malaysian audio** = whole zips downloaded (Xet) via `DownloadZipResolver`, read locally
  (≫ per-mp3 range request). Zips shared across shard workers (reused if already on disk).
  Bump `N_ZIPS` for big subsets (imda/dialects) if they're streaming-bound (low member hit-rate).
- **Malaysian re-run** (`deploy/pod_ms_redo.sh`): rebuild Malaysian only — deletes old `ms_*`
  shards, per subset predownloads a few *whole* zips then runs `CONC` sharded `pod_scale_big.py`
  workers in **`whole`+`sentence`** mode (`MS_MODE=whole MS_TURN_MODE=sentence TURN_GAP=1.5`),
  then `pod_ms_split.py` splits each subset's last shard 50/25/25 into `data/{train,validation,
  test}/`. Multilingual untouched. Verify with `deploy/pod_ms_verify.py` — it reports **tail RMS**
  (must be ≪ clip RMS = ends in silence). Loud-tailed turns are dropped, so Malaysian counts
  shrank (quality-over-quantity): e.g. dialects train 30 shards (was ~70).
- **Final dataset** `Scicom-intl/semantic-vad-eot`: 19 configs (`all`, 13 langs, 5 `ms_*`
  subsets), each `train`/`validation`/`test` (last shard split 50/25/25). Card = `pod_finalize.py`
  or a hand-uploaded README with `configs:` + `data_files` globs (`data/*.parquet`,
  `data/valid|train|test/...`); shard filenames don't match HF auto-detect, so the card is required.
- **Xet**: `pip install hf_xet` + `HF_XET_HIGH_PERFORMANCE=1` for fast HF transfer.
- Tar the repo with `COPYFILE_DISABLE=1 tar ... --exclude='._*'`; extract with `--no-same-owner`
  (macOS xattrs otherwise make GNU tar exit non-zero under `set -e`).
- SSH pattern (laptop → pod, tiny control traffic only):
  `ssh -i deploy/runpod_key -p <port> -o StrictHostKeyChecking=no root@<ip>`
- Run long jobs on the pod with `nohup ... > log 2>&1 &` and poll the log over SSH so a
  laptop disconnect doesn't kill them.
- Build outputs stay on the pod (`/root/data`). To deliver, push to HF Hub from the pod
  (HF_TOKEN is present) — don't download big parquet to the laptop.

## Testing

`.venv/bin/python -m pytest -q` (25 tests, offline, use real captured fixtures in
`tests/fixtures.py`; `test_training.py` covers the torch-free prompt/packing helpers). Keep
them network-free — the heavy training code (modeling/data/train) needs the `[train]` extra
+ a GPU and is exercised on the pod, not in `pytest`.

## Conventions

- eot-bench rule is positional: **do not store hold/eot labels**; keep `silence_spans`
  time-ordered with the EOT last.
- Turn mode by corpus: `single` for one-utterance rows (multilingual), `sentence` for
  punctuated recordings (Malaysian), `segment` only as a last resort (labels track gap size).
- **A turn's clip must end in silence** — `turn_to_row` VAD-trims the tail and drops turns
  that end loud. If you see a clip ending mid-word, check `speech_end_index` / `max_tail_ratio`.
- No heavyweight audio deps — `soundfile` covers wav+mp3; resampling is numpy linear.
