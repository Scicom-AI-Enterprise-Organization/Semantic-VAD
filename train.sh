WANDB_API_KEY=API_KEY \
WANDB_PROJECT=semantic-vad \
WANDB_NAME=eot-v1 \
torchrun --nproc_per_node 1 -m semvad.train \
    --output_dir runs/eot-v1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --dataset_name en --no-streaming \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 500 \
    --report_to wandb \
    --save_total_limit 5 \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 16 \
    --train_split "train[:20000]" \
    --eval_split "train[:100]" \
    --eval_strategy "steps" \
    --eval_steps 500

# after training, point eot-harness at the adapter this run wrote (must match --output_dir above):
#   EOT_CHECKPOINT_DIR=runs/eot-v1 eot-harness predict --path Scicom-intl/semantic-vad-eot \
#     --name en --split test --adapter semvad.eot_adapter:Qwen2AudioEoTAdapter --output-dir output