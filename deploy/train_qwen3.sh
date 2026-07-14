#!/usr/bin/env bash
# Runs ON a RunPod *GPU* pod (not the CPU dataset-build pod). Fine-tunes
# Whisper-encoder -> adapter -> Qwen3 for end-of-turn detection: full fine-tune (no LoRA),
# flash-attention varlen packing, optional frozen Whisper encoder.
#
# Prereqs on the pod:
#   uv pip install -e ".[train]"                          # torch, transformers>=4.51, accelerate, liger-kernel
#   uv pip install flash-attn --no-build-isolation        # needs CUDA + nvcc (matched to torch)
#   source /root/.hf_env                                   # HF_TOKEN (RunPod env not visible over SSH)
#   TRAIN_FILES points at eot parquet built by semantic_vad.build (pull from HF or /root/data)
set -euo pipefail

export PATH="/root/venv/bin:$PATH"
export HF_XET_HIGH_PERFORMANCE=1
[ -f /root/.hf_env ] && source /root/.hf_env

QWEN3_NAME="${QWEN3_NAME:-Qwen/Qwen3-0.6B}"     # small backbone -> fits the <1s serving budget
WHISPER_NAME="${WHISPER_NAME:-openai/whisper-base}"
TRAIN_FILES="${TRAIN_FILES:-/root/data/*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/out/eot-qwen3}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"        # set true to freeze the Whisper encoder
NPROC="${NPROC:-1}"                               # GPUs; torchrun handles >1

FREEZE_FLAG=""
[ "$FREEZE_ENCODER" = "true" ] && FREEZE_FLAG="--freeze_encoder"

# per_device_train_batch_size is how many examples get packed into ONE varlen sequence.
torchrun --nproc_per_node="$NPROC" -m semantic_vad.training.train \
  --qwen3_name "$QWEN3_NAME" \
  --whisper_name "$WHISPER_NAME" \
  --train_files "$TRAIN_FILES" \
  --output_dir "$OUTPUT_DIR" \
  --attn_implementation flash_attention_2 \
  $FREEZE_FLAG \
  --bf16 True \
  --block_size 4096 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --num_train_epochs 1 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --dataloader_num_workers 4 \
  --report_to none

echo "TRAIN_OK -> $OUTPUT_DIR"
