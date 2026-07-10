"""Fine-tune `Qwen2AudioEoTClassifier` on `Scicom-intl/semantic-vad-eot` with
`transformers.Trainer`.

Data parallelism (multi-GPU DDP) comes from `Trainer` + `accelerate`, not from
anything in this file -- launch with more than one process and it activates
automatically:

    torchrun --nproc_per_node=4 -m semvad.train \\
      --output_dir runs/eot-v1 --per_device_train_batch_size 8 \\
      --gradient_accumulation_steps 4 --bf16 --max_steps 20000 \\
      --dataset_name en --learning_rate 2e-4

or equivalently `accelerate launch -m semvad.train ...`. For a quick single-GPU/CPU
smoke run:

    python -m semvad.train --output_dir /tmp/eot-smoke --dataset_name en \\
      --streaming --max_steps 20 --per_device_train_batch_size 2 \\
      --logging_steps 1 --save_steps 20 --report_to none

For memory-constrained multi-GPU setups where the frozen 7B backbone doesn't fit
replicated on every rank, pass a DeepSpeed ZeRO-2 config via `--deepspeed
ds_config.json` (ZeRO-2 shards optimizer state/gradients, not parameters, so
`EoTTrainer._save` below stays correct; ZeRO-3 / full FSDP parameter sharding is
NOT supported by the checkpoint-saving override here -- see its docstring).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from semvad.data import EoTCollator, load_causal_dataset
from semvad.degrade import TelephonyDegrader
from semvad.metrics import compute_classification_metrics
from semvad.modeling import DEFAULT_MODEL_NAME, EoTHeadConfig, Qwen2AudioEoTClassifier

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelArguments:
    model_name_or_path: str = dataclasses.field(default=DEFAULT_MODEL_NAME)
    freeze_audio_tower: bool = dataclasses.field(default=True)
    head_hidden_size: int = dataclasses.field(default=256)
    head_dropout: float = dataclasses.field(default=0.1)
    use_lora: bool = dataclasses.field(default=True)
    lora_r: int = dataclasses.field(default=16)
    lora_alpha: int = dataclasses.field(default=32)
    lora_dropout: float = dataclasses.field(default=0.05)
    lora_target_modules: str = dataclasses.field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "comma-separated module names to attach LoRA adapters to"},
    )
    resume_adapter: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "dir written by a previous save_adapter() call to continue training from"}
    )
    attn_implementation: str = dataclasses.field(
        default="sdpa",
        metadata={"help": "attention backend passed to from_pretrained, e.g. sdpa/eager/flash_attention_2"},
    )


@dataclasses.dataclass
class DataArguments:
    dataset_path: str = dataclasses.field(default="Scicom-intl/semantic-vad-eot")
    dataset_name: Optional[str] = dataclasses.field(
        default="en", metadata={"help": "HF dataset config, e.g. en/de/es/... or 'all'"}
    )
    eval_dataset_path: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "defaults to --dataset_path; set to evaluate against a different dataset"},
    )
    eval_dataset_name: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "defaults to --dataset_name; config for --eval_dataset_path"}
    )
    train_split: str = dataclasses.field(default="train")
    eval_split: Optional[str] = dataclasses.field(default="validation")
    streaming: bool = dataclasses.field(
        default=False, metadata={"help": "stream from the hub instead of downloading the full config up front"}
    )
    eval_streaming: bool = dataclasses.field(
        default=False, metadata={"help": "stream from the hub instead of downloading the full config up front"} 
    )
    max_audio_seconds: float = dataclasses.field(default=16.0)
    max_eval_examples: int = dataclasses.field(
        default=512, metadata={"help": "cap eval-set size -- the full validation split is itself ~78k turns"}
    )
    dataloader_num_workers_data: int = dataclasses.field(
        default=0, metadata={"help": "num_proc for non-streaming .map() during example expansion"}
    )
    train_singleton_keep_prob: float = dataclasses.field(
        default=1.0,
        metadata={
            "help": (
                "keep-probability for TRAIN turns with exactly one silence span (eot-only, "
                "no hold). These turns are the majority and, because example weight is "
                "1/n_spans per turn, push the effective eot:hold loss-weight ratio to ~75:25 "
                "-- well past the raw span-count split. Lower this (e.g. 0.15-0.3) to correct "
                "the imbalance; 1.0 (default) disables downsampling. Never applied to eval."
            )
        },
    )
    telephony_augment: bool = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "simulate a call-centre telephony channel (narrowband + GSM/mu-law codec + "
                "line noise, see semvad/degrade.py) on TRAIN audio -- matches the deployment "
                "channel for this model. Never applied to eval."
            )
        },
    )
    telephony_apply_prob: float = dataclasses.field(
        default=0.7, metadata={"help": "fraction of train turns the telephony channel is applied to"}
    )
    telephony_packet_loss_prob: float = dataclasses.field(
        default=0.0, metadata={"help": "probability of additionally zeroing a few short VoIP-dropout chunks"}
    )


class EoTTrainer(Trainer):
    """Overrides checkpoint saving so we persist the ~1M-param head + LoRA adapter
    instead of a full 7B state dict on every `save_steps`.

    Correct for the common cases this script targets: single process, DDP, and
    DeepSpeed ZeRO-2 (parameters are not sharded in those, so `self.model` always
    holds real, complete weights on the saving rank -- see the comment at
    `Trainer._save_checkpoint` in the transformers source). NOT correct for
    DeepSpeed ZeRO-3 or FSDP full parameter sharding, which partition parameters
    across ranks and require `accelerator.get_state_dict(...)`-style gathering
    that this override skips; don't combine `--deepspeed <zero3 config>` or FSDP
    full sharding with this trainer without extending `_save` accordingly.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_adapter(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        logger.info("Saved EoT adapter (head + LoRA, not the frozen backbone) to %s", output_dir)


def preprocess_logits_for_metrics(logits, _labels):
    # `logits` here is `EoTOutput.to_tuple()` with `loss` stripped: (logits, p_eot).
    # Keep only the raw logit so `compute_metrics` doesn't have to juggle a tuple.
    if isinstance(logits, tuple):
        return logits[0]
    return logits


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return compute_classification_metrics(probs, labels)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Required: the raw dataset's columns (`audio`, `label`, `weight`, ...) don't
    # match `Qwen2AudioEoTClassifier.forward`'s parameter names -- only the
    # collator's *output* does. Trainer would otherwise drop them before the
    # collator ever sees them.
    training_args.remove_unused_columns = False
    if not training_args.label_names:
        training_args.label_names = ["labels"]

    if data_args.streaming and training_args.max_steps <= 0:
        raise ValueError(
            "--streaming datasets have no known length; set --max_steps explicitly "
            "instead of relying on --num_train_epochs."
        )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = Qwen2AudioEoTClassifier.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        freeze_audio_tower=model_args.freeze_audio_tower,
        head_config=EoTHeadConfig(head_hidden_size=model_args.head_hidden_size, dropout=model_args.head_dropout),
        attn_implementation=model_args.attn_implementation,
    )
    if model_args.resume_adapter:
        # Load the saved LoRA config + weights directly onto the plain backbone
        # instead of calling `apply_lora()` first -- see `load_adapter`'s
        # docstring for why stacking a fresh adapter under a resumed one is wrong.
        logger.info("Resuming training from adapter checkpoint %s", model_args.resume_adapter)
        model.load_adapter(model_args.resume_adapter, is_trainable=True)
    elif model_args.use_lora:
        model.apply_lora(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[m.strip() for m in model_args.lora_target_modules.split(",") if m.strip()],
        )

    n_total = model.count_parameters()
    n_trainable = model.count_parameters(trainable_only=True)
    logger.info("trainable params: %s / %s (%.3f%%)", f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total)

    degrader = None
    if data_args.telephony_augment:
        degrader = TelephonyDegrader(
            apply_prob=data_args.telephony_apply_prob,
            packet_loss_prob=data_args.telephony_packet_loss_prob,
        )
        logger.info(
            "Telephony channel augmentation enabled: apply_prob=%.2f packet_loss_prob=%.2f",
            data_args.telephony_apply_prob,
            data_args.telephony_packet_loss_prob,
        )

    train_dataset = load_causal_dataset(
        data_args.dataset_path,
        data_args.dataset_name,
        data_args.train_split,
        streaming=data_args.streaming,
        num_proc=data_args.dataloader_num_workers_data or None,
        singleton_keep_prob=data_args.train_singleton_keep_prob,
        degrader=degrader,
    )

    eval_dataset = None
    if data_args.eval_split:
        eval_dataset = load_causal_dataset(
            data_args.eval_dataset_path or data_args.dataset_path,
            data_args.eval_dataset_name or data_args.dataset_name,
            data_args.eval_split,
            streaming=data_args.eval_streaming,
            num_proc=data_args.dataloader_num_workers_data or None,
        )
        if data_args.max_eval_examples:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_examples)
            else:
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_examples)))

    collator = EoTCollator(processor, max_audio_seconds=data_args.max_audio_seconds)

    trainer = EoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset is not None else None,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
