# Semantic VAD

Audio-native End-of-Turn (EoT) detection: fine-tune **Qwen2-Audio-7B-Instruct** to predict
`p(eot)` directly from raw speech (no ASR transcript in the loop), and evaluate it with the
same causal, policy-sweep methodology LiveKit uses in
[eot-bench](https://github.com/livekit/eot-bench) /
["Solving end-of-turn detection"](https://livekit.com/blog/solving-end-of-turn-detection).

This document is the strategy/plan. Nothing below has been implemented yet — it's the spec
we'll build against.

## 1. Objective

- Take raw audio directly as model input (no separate STT step) and predict, at each pause,
  whether the user's turn has ended: `p(eot) ∈ [0, 1]`.
- The audio encoder embeds the speech and feeds it into the LLM as embeddings (Qwen2-Audio's
  native audio-tower → adapter → LLM path); the LLM head is fine-tuned to score "has the user
  finished speaking" rather than to converse.
- Base model: [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
  (Apache-2.0). Audio tower is a Whisper-large-v3-style encoder; a multimodal adapter projects
  audio features into the Qwen2-7B token-embedding space, so audio and text share one sequence.
- Training data: [`Scicom-intl/semantic-vad-eot`](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot).
- Evaluation: reproduce LiveKit's evaluation model — causal span-level scoring, policy sweep
  over `(threshold, action_delay, timeout)`, false-cutoff-rate vs. latency tradeoff, Pareto
  frontier — using their open-sourced harness, [`livekit/eot-bench`](https://github.com/livekit/eot-bench).

## 2. Problem framing

EoT is **not** offline classification over isolated clips. Both our training dataset and
LiveKit's benchmark encode the same structure, which we adopt as the task definition:

- Each row is one complete user turn, with `silence_spans`: every pause ≥100ms inside/after the
  turn.
- The **last** span in a turn is the true end of turn → label `eot`. Every earlier span is a
  mid-turn hesitation → label `hold`.
- A model must be queryable **causally**: at decision time `t` it may only see audio (and
  transcript, if used) up to `t`. It has no access to what the speaker says after `t`.
- The thing we ship is a per-span score `p(eot)`, not a chat response. Downstream, a serving
  policy (`threshold` / `action_delay` / `timeout`) turns that score into an actual "respond
  now" decision — same three knobs eot-bench sweeps.

This is exactly the schema `eot-bench` requires (`id`, `language`, `audio`, `silence_spans`,
optionally `messages`/`words`) — confirmed below, our training set matches it field-for-field,
which we treat as a first-class asset (see §5).

## 3. Data: `Scicom-intl/semantic-vad-eot`

Verified via the HF datasets-server API (not just the dataset card):

| Field | Type | Notes |
|---|---|---|
| `id` | string | `{lang}__{source}__turn_{n}` |
| `audio` | `Audio`, 16kHz | full-turn waveform |
| `language` | string | config/split language code |
| `duration` | float64 | seconds |
| `silence_spans` | list of `{start, end}` | ≥1 per row; **last = eot, rest = hold** |
| `words` | list of `{word, start, end}` | word-level forced-alignment timestamps |
| `messages` | list of `{role, content}` | always exactly 1 row: the user-turn transcript |

Scale: **7.54M train / 78.2k validation / 78.2k test** rows across the `all` config; per-language
configs (`en`, `de`, `es`, `fr`, `it`, `ja`, `ko`, `pl`, `pt`, `ru`, `th`, `tr`, `zh`, plus five
Malaysian-specific configs: `ms_dialects`, `ms_imda`, `ms_malaysian`, `ms_parliament`,
`ms_science_english`) are also independently loadable. Sampling 80 validation rows: ~46% of
turns have >1 silence span (i.e. real mid-turn hesitations are present, not just single-pause
utterances), median duration ~3-4s, `messages` is always a single user turn with no prior
assistant context — consistent with the "current-turn-only, no long chat history" design goal
described in LiveKit's blog post. License: CC-BY-4.0.

**Language overlap with `livekit/eot-bench-data`** (`{ar, de, en, es, fr, hi, id, it, ja, ko,
nl, pt, tr, zh}`, 14 languages): we have direct training data for **10 of the 14**
(`de, en, es, fr, it, ja, ko, pt, tr, zh`). We have **no data** for `ar`, `hi`, `id`, `nl` — any
cross-benchmark numbers on those four are zero-shot and out of scope for v1. Conversely we have
substantial Malaysian-market coverage (`pl`, `ru`, `th`, and the five `ms_*` configs) that
eot-bench doesn't test at all — worth reporting separately as a differentiator, not a benchmark
comparison.

### Causal example construction (the part that's easy to get subtly wrong)

Training examples must be built to match how the model will be *queried* at eval/serving time,
or we train/test-mismatch on the one thing that matters most:

1. For each row, take `silence_spans` sorted by `start`. Label the last one `eot`, the rest
   `hold`.
2. For each span, generate one or more **causal truncation points**: cut the waveform at
   `span.start + Δ` for `Δ ∈ {0.0, 0.2, 0.6, 1.2}s` (clipped to `span.end`, and to the row's
   post-span audio for `eot` spans, which have none). This teaches the model to produce
   *monotonically increasing* confidence the longer a `hold` silence lasts, which is what the
   harness's grid-scoring (every `inference_interval`, default 100ms, across the whole span)
   actually probes at eval time — a model trained only on `Δ=0` will be well-calibrated at the
   instant silence starts and untested (and likely miscalibrated) 800ms into it.
3. Drop the audio after the cut point entirely (do not just mask it — Qwen2-Audio's encoder
   sees the raw waveform, so leakage has to be prevented at the audio level, not the token
   level).
4. Carry the `messages`/`words` fields, trimmed the same way eot-bench trims them (append a
   user-message fragment of words whose `end ≤ cut_time − transcript_lag`, default lag 0.5s) —
   *only if* the chosen architecture uses text context at all (see §4).

### Class balance / sampling

Per-row, `eot:hold` is roughly 1:1–1:2 (since ~46% of rows contribute extra `hold` spans). We'll
weight examples by `1/num_spans_in_row` so no single multi-hesitation row dominates the `hold`
class, and stratify sampling across languages so low-resource configs aren't drowned out by the
500k-row majors.

## 4. Model & task head

Two ways to get `p(eot)` out of Qwen2-Audio; we build the simpler one first and only reach for
the second if it's not good enough.

### Option A — next-token probability (v1, build this first)

Fine-tune the stock `Qwen2AudioForConditionalGeneration` with the standard causal-LM loss,
masked so gradient only flows to one label token. Prompt template: audio input followed by a
fixed instruction ("Has the speaker finished their turn? Answer yes or no."); target token is
`Yes`/`No`. At inference, **one forward pass** (no autoregressive decoding) reading the softmax
over the `{Yes, No}` token ids at the final position gives `p(eot)`.

- Reuses the stock HF training loop, LoRA/QLoRA, and generation utilities unchanged — fastest
  path to a working model.
- Matches the literal framing in this repo's original brief ("the model predicts the token [that]
  is it the end of speech") and precedent from LiveKit's own earlier open turn-detector, which
  used the same token-probability trick.
- Downsides: sensitive to prompt phrasing/tokenization, and technically pays for a text prompt
  in the context even though we never decode past one token.

### Option B — dedicated classification head (fast-follow, if A's latency/robustness isn't enough)

Wrap the Qwen2-Audio backbone (audio tower + adapter + LLM trunk, no LM head) with a single
linear layer over the pooled final hidden state → one logit → `BCEWithLogitsLoss`. No prompt
template, no vocabulary dependency, marginally cheaper per call, and a closer match to how
`eot_harness`'s batch adapter contract wants a score (`predict_batch` returns a bare
`list[float]`) and to how LiveKit describes their production "semantic branch."

We start with **Option A** for milestone 1 (§7) because it's a strict subset of standard
fine-tuning code, then benchmark whether Option B is worth the extra plumbing once we have real
harness numbers to compare against.

Audio tower: freeze initially (preserve pretrained acoustic representations); fine-tune the
multimodal adapter + LoRA over the LLM trunk. Revisit unfreezing the top audio-tower layers only
if error analysis shows acoustic/prosodic cues (not semantics) are the bottleneck.

## 5. Evaluation strategy

We evaluate two ways, both through LiveKit's own harness so the methodology is identical to
their published results — not a reimplementation we have to defend separately.

**5.1 — Internal: our own held-out test split.** Because `Scicom-intl/semantic-vad-eot` already
satisfies `eot_harness`'s dataset contract (`id`, `language`, `audio`, `silence_spans` required;
`messages`, `words` optional — verified against `eot_harness/schemas.py`), we can point the
harness at it directly, no adapter-side dataset shimming needed:

```bash
PYTHONPATH=. EOT_CHECKPOINT_DIR=runs/eot-v1 \
eot-harness predict --path Scicom-intl/semantic-vad-eot --name en --split test \
  --adapter semvad.eot_adapter:Qwen2AudioEoTAdapter --output-dir output
eot-harness compute-metrics --predictions output/.../predictions.parquet --output-dir output/.../metrics
eot-harness compare-models output/scicom-intl__semantic-vad-eot__test__min_silence_100ms/en
```

`EOT_CHECKPOINT_DIR` (or the adapter's `checkpoint_dir` constructor arg) must point at the dir a training
run's `--output_dir` wrote (see `train.sh`) -- without it, `Qwen2AudioEoTAdapter` loads the base model with
a randomly-initialized head and scores are meaningless.

Run per-language (`--name de|es|fr|...`) and roll up with `compare-languages`. This is our fast
inner-loop signal during training iteration.

**5.2 — External: `livekit/eot-bench-data`.** Same adapter, pointed at LiveKit's benchmark
dataset, gives numbers directly comparable to their published leaderboard table (LiveKit v1:
9.9% false-cutoffs @300ms / 543ms latency @5% cutoff; Deepgram Flux 12.9% / 1151ms; ultraVAD
27.7%; AssemblyAI 49.4%; VAD-only baseline 55.6%, etc.) on the 10 overlapping languages. This is
the "apples-to-apples vs. published results" check.

**Metrics** (per the harness's own stated philosophy — scalar `auc`/`ap` are diagnostic only and
don't drive rankings):
- False-cutoff rate at fixed latency budgets (300ms, 600ms).
- Mean latency at fixed false-cutoff budgets (5%, 10%).
- Full false-cutoff/latency Pareto frontier.
- Per-language heatmaps via `compare-languages`.
- A VAD-only baseline is computed automatically by the harness on the same policy grid — our
  floor to always beat.

**Adapter implementation** (`semvad.eot_adapter:Qwen2AudioEoTAdapter`): implements
`adapter_id`, `score_point`, and `predict_batch(batch)`; each batch item's `audio` is already the
harness-sliced causal prefix, so no truncation logic lives in the adapter — only preprocessing
into Qwen2-Audio's processor format and reading `p(eot)` off the model per §4. `supports_language`
restricted to the languages we actually train on, so unsupported-language rows are skipped
rather than silently mis-scored.

## 6. Training infrastructure

- 7.5M rows / ~150GB audio in the `all` config — too large to fully materialize as fixed
  causal clips up front at full scale. Plan: stream via `datasets` (`Audio(decode=False)`, decode
  + truncate on the fly in the collator, mirroring how the harness itself avoids `torchcodec`).
  Pre-extract a fixed, cached subset for milestone 1 so iteration isn't bottlenecked on
  streaming I/O.
- PEFT: LoRA (or QLoRA if a single 24–48GB GPU is the constraint) over the LLM trunk; audio
  tower frozen (§4). bf16, gradient checkpointing, Flash-Attention 2.
- Start with a stratified subset (a few hundred k examples across languages + truncation
  offsets) to get the training/eval loop correct before scaling to the full multilingual mixture.

## 7. Milestones

1. **M0 – Pipeline**: causal span-extraction + truncation script; small English subset; sanity
   fine-tune to confirm the loss goes down and the adapter round-trips through `eot-harness`
   end-to-end (predict → compute-metrics → compare-models) on a toy run.
2. **M1 – Baseline (Option A)**: LoRA fine-tune on `en` + a few major languages; first real
   internal-test-split harness numbers (§5.1); establish whether we beat the VAD-only baseline
   and by how much.
3. **M2 – Multilingual scale-up**: extend to all 10 eot-bench-overlapping languages + the
   Malaysian-specific configs; class-balance/sampling tuning; run the full external comparison
   against `livekit/eot-bench-data` (§5.2) for a leaderboard-style number.
4. **M3 – Iterate**: use `compare-models`/`compare-languages` reports (Pareto plots, per-language
   heatmaps) to find failure modes — early false cutoffs vs. laggy true-EoT detection — and
   decide whether Option B (classification head), unfreezing the audio tower, or more
   multi-offset training data closes the gap to LiveKit v1's numbers.
5. **M4 – Stretch**: distill to a smaller model for a "v1-mini" latency tier; consider a fused
   acoustic branch (raw silence-timing signal, LiveKit's dual-branch design) if semantic-only
   plateaus above the VAD floor on ambiguous pauses.

## 8. Open questions / risks

- **Language coverage gap**: no training data for `ar`, `hi`, `id`, `nl` — any eot-bench numbers
  there are zero-shot, not a fair comparison. Flagging so it's not read as an apples-to-apples
  result across all 14 languages.
- **Option A vs. B**: starting with token-probability (A) because it reuses stock fine-tuning
  code; the classification-head route (B) is a deliberate fast-follow, not a fallback we're
  avoiding — will revisit once M1 numbers exist.
- **Repo currently has a `gemma/` directory** (a downloaded Gemma-4 omni checkpoint with
  audio/vision token ids) that doesn't match this plan's stated base model (Qwen2-Audio-7B).
  Left untouched — flagging in case it's leftover from earlier exploration rather than intentional.
