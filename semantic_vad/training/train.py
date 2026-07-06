"""Fine-tune Whisper-encoder → adapter → Qwen3 for end-of-turn detection.

Full fine-tune (no LoRA), flash-attention **varlen** packing, optional frozen Whisper
encoder. Run on a **GPU** pod (not the CPU dataset-build pod) with the ``train`` extra
installed plus a flash-attention build. See ``deploy/train_qwen3.sh`` for a launch command.

Example (single GPU)::

    python -m semantic_vad.training.train \\
        --qwen3_name Qwen/Qwen3-0.6B \\
        --whisper_name openai/whisper-base \\
        --train_files "data/*.parquet" \\
        --output_dir out/eot-qwen3 \\
        --bf16 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \\
        --learning_rate 1e-5 --warmup_ratio 0.03 --num_train_epochs 1 \\
        --gradient_checkpointing --logging_steps 10 --save_steps 500
"""

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperModel,
    set_seed,
)

from .data import SemanticVADDataset, collate_packed
from .modeling import WhisperQwen3
from .prompt import (
    AUDIO_TOKEN,
    DEFAULT_SYSTEM,
    EOT_TOKEN,
    HOLD_TOKEN,
    SPECIAL_TOKENS,
    SpecialIds,
)


@dataclass
class ModelArguments:
    qwen3_name: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Qwen3 causal-LM checkpoint (the text backbone)."},
    )
    whisper_name: str = field(
        default="openai/whisper-base",
        metadata={"help": "Whisper checkpoint whose encoder + config are used for audio."},
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Freeze the Whisper encoder (projection + Qwen3 stay trainable)."}
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "flash_attention_2 (varlen) or an fa3 kernel id e.g. "
                          "'kernels-community/vllm-flash-attn3'. Do NOT use sdpa/eager (no varlen)."},
    )
    torch_dtype: str = field(default="bfloat16", metadata={"help": "auto|bfloat16|float16|float32"})


@dataclass
class DataArguments:
    train_files: str = field(
        default=None,
        metadata={"help": "Parquet path or glob (e.g. 'data/*.parquet') of the eot dataset."},
    )
    block_size: int = field(default=4096, metadata={"help": "Max tokens per example; longer examples are dropped."})
    include_holds: bool = field(
        default=True, metadata={"help": "Also emit <|hold|> examples truncated at each mid-turn pause."}
    )
    supervise_transcript: bool = field(
        default=True,
        metadata={"help": "Add the transcript after the marker as adapter-alignment supervision."},
    )
    system_prompt: str = field(default=DEFAULT_SYSTEM, metadata={"help": "System prompt text."})


def _resolve_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet files matched --train_files {pattern!r}")
    return files


def main(argv: Optional[list[str]] = None) -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)
    if data_args.train_files is None:
        raise SystemExit("--train_files is required")
    set_seed(training_args.seed)

    dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ("auto", None)
        else getattr(torch, model_args.torch_dtype)
    )

    # Liger kernels for Qwen3 (rope/rmsnorm/swiglu); CE is handled by our fused-linear loss.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, rms_norm=True, cross_entropy=False,
            fused_linear_cross_entropy=False,
        )
    except Exception:
        pass

    # -- tokenizer: add audio placeholders + EOT/HOLD markers --------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    special_ids = SpecialIds(
        im_start=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        im_end=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        audio_bos=tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        audio_eos=tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )
    marker_ids = {
        "eot": tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        "hold": tokenizer.convert_tokens_to_ids(HOLD_TOKEN),
    }

    # -- compose the config (Qwen3 text + Whisper encoder + audio/marker ids) ----------
    config = AutoConfig.from_pretrained(model_args.qwen3_name)
    whisper_config = AutoConfig.from_pretrained(model_args.whisper_name)
    # Whisper checkpoints are encoder-decoder; keep just the encoder-relevant config.
    encoder_config = getattr(whisper_config, "encoder", None) or whisper_config
    config.audio_encoder_config = encoder_config.to_dict()  # dict → JSON-serializable
    config.audio_token_index = special_ids.audio
    config.eot_token_index = marker_ids["eot"]
    config.hold_token_index = marker_ids["hold"]
    config.use_cache = False

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_name)

    # -- build model: Qwen3 weights + freshly-injected pretrained Whisper encoder ------
    model = WhisperQwen3.from_pretrained(
        model_args.qwen3_name,
        config=config,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
    )
    whisper = WhisperModel.from_pretrained(model_args.whisper_name, torch_dtype=dtype)
    model.encoder.load_state_dict(whisper.get_encoder().state_dict())
    del whisper
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    if model_args.freeze_encoder:
        model.freeze_encoder()
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()  # needed when inputs come in as embeds

    dataset = SemanticVADDataset(
        _resolve_files(data_args.train_files),
        tokenizer,
        feature_extractor,
        special_ids,
        marker_ids,
        block_size=data_args.block_size,
        include_holds=data_args.include_holds,
        supervise_transcript=data_args.supervise_transcript,
        system=data_args.system_prompt,
    )
    print(f"examples: {len(dataset)} (from {len(dataset.ds)} turn rows)", flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_packed,
        processing_class=tokenizer,
    )

    checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()
    feature_extractor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
