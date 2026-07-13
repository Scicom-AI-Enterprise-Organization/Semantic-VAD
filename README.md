# Semantic VAD

Audio-native end-of-turn (EoT) detection: a fine-tuned **Qwen2-Audio-7B-Instruct** that predicts
`p(eot)` directly from raw speech at each pause in a conversation — no ASR transcript, no
separate turn-detection model in the loop. Evaluated with the same causal, policy-sweep
methodology LiveKit uses in [`eot-bench`](https://github.com/livekit/eot-bench) /
["Solving end-of-turn detection"](https://livekit.com/blog/solving-end-of-turn-detection).

See [`PLAN.md`](PLAN.md) for the original design doc (problem framing, data schema, milestones).
This document describes the system as built.

## How it works

- **Backbone**: [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
  (Apache-2.0), loaded as `Qwen2AudioModel` — audio tower + multimodal adapter + LLM trunk, with
  the 640M-parameter `lm_head` dropped entirely (see `semvad/modeling.py`).
- **Head**: a ~1M-parameter binary classification head (`LayerNorm → Linear → GELU → Linear → 1
  logit`) on the pooled last hidden state. One forward pass → `p(eot) = sigmoid(logit)`. No
  autoregressive decoding, no vocabulary projection.
- **Fine-tuning**: audio tower frozen; LoRA over the LLM trunk (`q/k/v/o_proj`,
  `gate/up/down_proj`); the head trains from scratch. Only the head + LoRA weights are
  checkpointed (`save_adapter`/`load_adapter` in `semvad/modeling.py`), not the frozen 7B
  backbone.
- **Causal training data**: each row of
  [`Scicom-intl/semantic-vad-eot`](https://huggingface.co/datasets/Scicom-intl/semantic-vad-eot)
  is one user turn with `silence_spans` — the last span is the true `eot`, every earlier span is
  a mid-turn `hold`. Each span is cut at multiple offsets (`0.0/0.2/0.6/1.2s` into the pause, see
  `semvad/data.py`) so the model learns confidence should rise the longer a pause lasts, matching
  how `eot-bench` scores every `inference_interval` across a span rather than just its start.
- **Telephony augmentation**: an optional training-time channel simulator (`semvad/degrade.py`) —
  narrowband filtering, GSM/µ-law codec round-trip, line noise, packet-loss dropouts — so the
  model generalizes to a call-centre deployment channel, not just clean studio audio. **Planned,
  not yet trained**: `--telephony_augment` is implemented and wired into `semvad/train.py`, but
  the `eot-v6` checkpoint below was trained without it (clean audio only). The results below say
  nothing about telephony-channel robustness yet — that's a follow-up training run, not a
  reflection of the augmentation not working.

## Results

All numbers below are for `eot-v6/checkpoint-2000`, trained on clean (non-telephony-augmented)
audio only — see the telephony augmentation note above. Robustness on a degraded call-centre
channel is untested until a checkpoint is trained with `--telephony_augment`.

### Classification (`Scicom-intl/semantic-vad-eot`, English, checkpoint `eot-v6/checkpoint-2000`)

500 turns sampled with `--seed 42` from the `test` split:

| language | n    | accuracy | f1    | auc   |
|----------|------|----------|-------|-------|
| en       | 2674 | 0.840    | 0.861 | 0.923 |
| overall  | 2674 | 0.840    | 0.861 | 0.923 |

Same sample scored against LiveKit's cloud `turn-detector-v1` for comparison:

| language | n    | accuracy | f1    | auc   |
|----------|------|----------|-------|-------|
| en       | 2674 | 0.643    | 0.723 | 0.671 |
| overall  | 2674 | 0.643    | 0.723 | 0.671 |

### Cross-benchmark (`livekit/eot-bench-data`, English, `validation` split, 400 turns / seed 42)

This dataset's turns skew differently from our own held-out split, so both backends are
re-scored on it for a fair apples-to-apples read:

| backend         | n    | accuracy | f1    | auc   |
|-----------------|------|----------|-------|-------|
| ours (local)    | 4026 | 0.690    | 0.676 | 0.808 |
| LiveKit v1 cloud| 4026 | 0.858    | 0.843 | 0.939 |

Our model leads on our own dataset's distribution but trails LiveKit v1 on their benchmark's
distribution. This isn't just "their model is tuned on their data" — the two datasets encode the
`hold`/`eot` decision itself differently, which [`dataset_analysis.ipynb`](dataset_analysis.ipynb)
quantifies by applying the same causal labeling rule (last span per turn = `eot`, earlier spans =
`hold`) to 200-row samples of each and comparing span counts/durations:

| dataset                          | hold spans | eot spans | turns with >1 span | eot span length |
|-----------------------------------|-----------:|----------:|--------------------:|------------------|
| `Scicom-intl/semantic-vad-eot`    | 13.0%      | 87.0%     | 28 / 200 (14%)       | fixed at 0.500s  |
| `livekit/eot-bench-data`          | 66.3%      | 33.7%     | 108 / 200 (54%)      | fixed at 1.500s  |

Our training set is dominated by short, single-pause turns with the true end-of-turn always cut
at a fixed 0.5s offset and `hold` pauses averaging 0.27s. `eot-bench-data` is the opposite: most
turns carry several mid-turn hesitations (up to 14 in the sample), `hold` pauses run longer and
more variable (mean 0.43s, up to 3.3s), and the `eot` cut point is fixed at 1.5s instead of 0.5s.
A model trained on the first distribution and evaluated on the second (or vice versa) is being
asked to generalize across a genuinely different pause-duration prior, not just a different
recording domain — which is why the local-vs-LiveKit ranking flips between the two benchmark
tables above. Closing this gap is the open work tracked in [`PLAN.md`](PLAN.md) (§7–8:
multilingual scale-up, Option B classification head, error analysis).

### Latency (single causal forward pass, RTX 3090, bf16, `eot-v6/checkpoint-2000`)

`scripts/benchmark_latency.py`, 20 repeats per duration after 3 warmup calls:

| audio duration | mean (ms) | p50 (ms) | p95 (ms) |
|---------------:|----------:|---------:|---------:|
| 0.5s            | 115.8     | 112.8    | 125.9    |
| 1s              | 118.1     | 113.5    | 124.6    |
| 2s              | 120.8     | 114.6    | 166.7    |
| 4s              | 121.6     | 115.0    | 156.2    |
| 8s              | 159.6     | 159.3    | 161.8    |
| 16s             | 233.6     | 232.8    | 238.3    |
| 24s             | 260.6     | 260.4    | 261.9    |
| 30s             | 336.5     | 335.9    | 339.0    |

Qwen2-Audio's feature extractor pads every clip to a fixed 30s mel grid regardless of actual
length, so audio-tower cost stays roughly flat; latency scales with the LLM trunk's attention
over the growing audio-token sequence. Sub-4s pauses (the common case at decision time) score in
~115-120ms on a 3090 — well inside typical `action_delay`/`timeout` policy budgets.

## Repo layout

```
semvad/
  modeling.py     Qwen2AudioEoTClassifier — backbone + head, LoRA, checkpointing, predict_p_eot
  data.py         causal example construction (silence-span cutting) + collator
  degrade.py      telephony channel augmentation (codec/noise/packet-loss)
  train.py        HF Trainer fine-tuning entrypoint
  eot_adapter.py  eot-harness batch adapter (Qwen2AudioEoTAdapter)
  metrics.py      shared accuracy/f1/auc computation (train-time eval + benchmark script)
scripts/
  benchmark_eot.py       accuracy/f1/auc, --backend local (ours) or livekit (cloud turn-detector-v1)
  benchmark_latency.py   forward-pass latency across audio durations
  duration_bias_check.py checks whether p(eot) is biased by absolute clip duration
  inspect_errors.py      dumps misclassified examples for error analysis
  preview_telephony_degrade.py  listen to/inspect the augmentation pipeline's output
  smoke_test.py, train_smoke_test.py  fast sanity checks (no full training run)
app/
  gradio_app.py   live mic + file-upload demo, --mock (no GPU) or real checkpoint
train.sh          example multi-epoch LoRA training invocation
setup.sh          environment bootstrap (uv venv + deps + eot-harness)
```

## Setup

```bash
git clone https://github.com/tchiayan/sematic_vad.git
cd sematic_vad
bash setup.sh
source .venv/bin/activate
```

`setup.sh` creates a `uv`-managed Python 3.12 venv, installs `requirements.txt`, and installs
`eot-harness` from LiveKit's repo for benchmarking. Training and the full (non-`--mock`) demo
need a CUDA GPU — Qwen2-Audio-7B is a 7B-parameter audio-language model.

Download a trained checkpoint (or train your own, below):

```bash
hf download tchiayan/semantic-vad-v6 --repo-type dataset --local-dir .
unzip eot-v6.zip   # -> runs/eot-v6/checkpoint-2000 (eot_head.pt + lora/)
```

## Training

```bash
WANDB_PROJECT=semantic-vad WANDB_NAME=eot-v1 \
torchrun --nproc_per_node 1 -m semvad.train \
    --output_dir runs/eot-v1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --dataset_name en --no-streaming \
    --learning_rate 1e-4 \
    --logging_steps 1 --save_steps 500 --save_total_limit 5 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 16 \
    --train_split "train[:20000]" --eval_split "train[:100]" \
    --eval_strategy steps --eval_steps 500
```

See `train.sh` for the full example and `semvad/train.py --help` (via `HfArgumentParser`) for
every `ModelArguments`/`DataArguments`/`TrainingArguments` flag — including `--use_lora`,
`--lora_r`/`--lora_alpha`, `--telephony_augment`, `--resume_adapter`, and multi-GPU DDP
(`torchrun --nproc_per_node N ...`) or DeepSpeed ZeRO-2 (`--deepspeed <config>`; ZeRO-3/FSDP full
sharding is not supported by the checkpoint-saving override in `EoTTrainer._save`).

Each `save_steps` checkpoint under `--output_dir` contains only the trainable bits: `eot_head.pt`
+ a `lora/` adapter dir — not the frozen 7B backbone.

## Evaluating

### Accuracy / F1 / AUC

```bash
# our checkpoint, against our own held-out test split
PYTHONPATH=./ python scripts/benchmark_eot.py --backend local \
    --checkpoint runs/eot-v6/checkpoint-2000 \
    --dataset-name en --split test --limit 500 --seed 42 --output local_en.json

# LiveKit's cloud turn-detector-v1, same sample, for a fair comparison
export LIVEKIT_URL=wss://...  LIVEKIT_API_KEY=...  LIVEKIT_API_SECRET=...
PYTHONPATH=./ python scripts/benchmark_eot.py --backend livekit \
    --dataset-name en --split test --limit 500 --seed 42 --output livekit_en.json

# either backend against LiveKit's own benchmark dataset instead of ours
PYTHONPATH=./ python scripts/benchmark_eot.py --backend local \
    --checkpoint runs/eot-v6/checkpoint-2000 \
    --dataset-path livekit/eot-bench-data \
    --dataset-name en --split validation --limit 400 --seed 42 --output local_livekit.json
```

`--backend` swaps between our local checkpoint and LiveKit's hosted model; `--dataset-path`
swaps the dataset. Keep `--seed`/`--limit` fixed across runs being compared — rows are drawn via
the same shuffle+take convention regardless of backend, so any (`dataset-path`, `dataset-name`,
`split`, `limit`, `seed`) tuple always samples identical rows.

For the full LiveKit eot-bench methodology (policy sweep over `threshold`/`action_delay`/
`timeout`, Pareto frontier, per-language breakdown) instead of this repo's flat accuracy/f1/auc,
point `eot-harness` directly at the adapter:

```bash
PYTHONPATH=. EOT_CHECKPOINT_DIR=runs/eot-v6/checkpoint-2000 \
eot-harness predict --path Scicom-intl/semantic-vad-eot --name en --split test \
  --adapter semvad.eot_adapter:Qwen2AudioEoTAdapter --output-dir output
eot-harness compute-metrics --predictions output/.../predictions.parquet --output-dir output/.../metrics
eot-harness compare-models output/scicom-intl__semantic-vad-eot__test__min_silence_100ms/en
```

### Latency

```bash
PYTHONPATH=./ python scripts/benchmark_latency.py --device cuda \
    --checkpoint runs/eot-v6/checkpoint-2000 \
    --durations 0.5 1 2 4 8 16 24 30 --repeats 20 --output latency.json
```

Run on the GPU box you intend to deploy on — see [Results](#results) above for numbers on an
RTX 3090.

## Demo

```bash
# no GPU / no model download — heuristic trailing-silence predictor, for exercising the UI
python -m app.gradio_app --mock

# real model
python -m app.gradio_app --checkpoint runs/eot-v6/checkpoint-2000 --device cuda
```

Live microphone tab streams `p(eot)` as you speak and pause; the upload tab scores a full file
and marks each detected end-of-turn cutoff on the spectrogram. Sliders control the same three
policy knobs `eot-bench` sweeps (`threshold`, `action_delay`, `timeout`).
