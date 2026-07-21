"""Fine-tune `WhisperQwen3EoTClassifier` (Option C) on `Scicom-intl/semantic-vad-eot` with
`transformers.Trainer`.

Option C = Option B's backbone (Whisper encoder -> linear projection -> Qwen3, no native
multimodal checkpoint required) + Option A's head (`EoTHead` on the pooled last hidden
state, `semvad/modeling.py`) + a **full fine-tune** (no LoRA -- encoder, projection, Qwen3
trunk, and head are all trainable, same as Option B's philosophy, just with a classification
head instead of a marker-token read). Dataset pipeline (causal per-span truncation, class
weighting, telephony augmentation) is reused wholesale from `semvad/` -- see `data.py`.

No flash-attn varlen packing (unlike Option B): each example is one independent
classification label, so ordinary padded batching is the right shape, same as Option A.
Runs on a GPU pod (the `train` extra: torch, transformers>=4.51, accelerate) -- no
liger-kernel/flash-attn required (no fused-linear-CE vocab loss, no varlen kernel).

Example (single GPU)::

    torchrun --nproc_per_node=1 -m whisper_qwen3_head.train \\
      --output_dir runs/eot-whisper-qwen3-head --per_device_train_batch_size 16 \\
      --gradient_accumulation_steps 8 --bf16 --dataset_name en --no-streaming \\
      --learning_rate 5e-5 --logging_steps 10 --save_steps 500 --report_to none

For a quick single-GPU/CPU smoke run:

    python -m whisper_qwen3_head.train --output_dir /tmp/eot-smoke --dataset_name en \\
      --streaming --max_steps 20 --per_device_train_batch_size 2 \\
      --logging_steps 1 --save_steps 20 --report_to none

To push the finished checkpoint (encoder + backbone + projection + head + tokenizer +
feature extractor) to the Hub, add ``--push_to_hub --hub_model_id <org>/<name>`` (also
respects ``--hub_private_repo``/``--hub_token``/``--hub_revision``, all standard
`TrainingArguments` fields) -- this calls `WhisperQwen3EoTClassifier.push_to_hub` once
training finishes, not Trainer's own built-in hub sync (see the comment in `main()`).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments, WhisperFeatureExtractor

from semvad.degrade import TelephonyDegrader
from semvad.metrics import compute_classification_metrics
from semvad.modeling import EoTHeadConfig
from semvad.train import DataArguments  # identical dataset pipeline as Option A -- see semvad/data.py
from whisper_qwen3_head.data import EoTCollator, load_causal_dataset
from whisper_qwen3_head.modeling import DEFAULT_QWEN3_NAME, DEFAULT_WHISPER_NAME, WhisperQwen3EoTClassifier
from whisper_qwen3_head.prompt import AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN, SPECIAL_TOKENS, SpecialIds

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelArguments:
    qwen3_name: str = dataclasses.field(default=DEFAULT_QWEN3_NAME)
    whisper_name: str = dataclasses.field(default=DEFAULT_WHISPER_NAME)
    freeze_encoder: bool = dataclasses.field(
        default=False, metadata={"help": "freeze the Whisper encoder (projection + Qwen3 + head stay trainable)"}
    )
    head_hidden_size: int = dataclasses.field(default=256)
    head_dropout: float = dataclasses.field(default=0.1)
    resume_checkpoint: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "dir written by a previous WhisperQwen3EoTClassifier.save_pretrained() call to continue training from"},
    )
    attn_implementation: str = dataclasses.field(
        default="sdpa",
        metadata={"help": "attention backend for the Qwen3 backbone, e.g. sdpa/eager/flash_attention_2"},
    )


class EoTTrainer(Trainer):
    """Overrides checkpoint saving to call `WhisperQwen3EoTClassifier.save_pretrained`
    (separate encoder/backbone/projection/head pieces) instead of `Trainer`'s default, which
    -- since this model is a plain `nn.Module`, not a `PreTrainedModel` -- would otherwise
    just dump one raw `state_dict` blob that `from_checkpoint` can't reload.

    Unlike `semvad.train.EoTTrainer` (which persists only a ~1M-param head + LoRA adapter
    because the 7B backbone is frozen), this is a **full fine-tune**: everything is saved.
    """

    def __init__(self, *args, feature_extractor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(output_dir)
        logger.info("Saved WhisperQwen3EoTClassifier (encoder + backbone + projection + head) to %s", output_dir)


def preprocess_logits_for_metrics(logits, _labels):
    # `logits` here is `EoTOutput.to_tuple()` with `loss` stripped: (logits, p_eot).
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

    # Required: the raw dataset's columns (`audio`, `label`, `weight`, ...) don't match
    # `WhisperQwen3EoTClassifier.forward`'s parameter names -- only the collator's *output*
    # does. Trainer would otherwise drop them before the collator ever sees them.
    training_args.remove_unused_columns = False
    if not training_args.label_names:
        training_args.label_names = ["labels"]

    # `--push_to_hub` (+ `--hub_model_id`/`--hub_private_repo`/`--hub_token`/`--hub_revision`)
    # are standard `TrainingArguments` fields, reused here for the CLI surface -- but Trainer's
    # *own* built-in push (`Trainer.push_to_hub`, wired to fire from `save_model`/periodic
    # checkpoints when `args.push_to_hub` is left `True`) assumes `self.model` is a
    # `PreTrainedModel` (e.g. its `create_model_card` introspects `model.config`), which ours
    # isn't. Read the flag now, then disable it on `training_args` so Trainer's internal hub
    # machinery never triggers, and push explicitly via `model.push_to_hub` (this module's,
    # not Trainer's) once training has actually finished -- see the bottom of `main()`.
    push_to_hub = training_args.push_to_hub
    training_args.push_to_hub = False

    if data_args.streaming and training_args.max_steps <= 0:
        raise ValueError(
            "--streaming datasets have no known length; set --max_steps explicitly "
            "instead of relying on --num_train_epochs."
        )

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    tokenizer_source = model_args.resume_checkpoint or model_args.qwen3_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    special_ids = SpecialIds(
        audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
        audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )
    head_config = EoTHeadConfig(head_hidden_size=model_args.head_hidden_size, dropout=model_args.head_dropout)

    if model_args.resume_checkpoint:
        logger.info("Resuming model weights from %s", model_args.resume_checkpoint)
        model = WhisperQwen3EoTClassifier.from_checkpoint(
            model_args.resume_checkpoint,
            dtype=dtype,
            attn_implementation=model_args.attn_implementation,
            head_config=head_config,
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.resume_checkpoint)
    else:
        model = WhisperQwen3EoTClassifier.from_pretrained(
            model_args.qwen3_name,
            model_args.whisper_name,
            dtype=dtype,
            freeze_encoder=model_args.freeze_encoder,
            head_config=head_config,
            attn_implementation=model_args.attn_implementation,
        )
        # Built here (not inside `from_pretrained`, which stays tokenizer-agnostic) since the
        # target vocab size depends on the special tokens just added above.
        model.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_name)
    model.set_special_ids(special_ids)

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
        num_proc=data_args.dataloader_num_workers_data,
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
            num_proc=data_args.dataloader_num_workers_data,
        )
        if data_args.max_eval_examples:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_examples)
            else:
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_examples)))

    collator = EoTCollator(tokenizer, feature_extractor, special_ids, max_audio_seconds=data_args.max_audio_seconds)

    trainer = EoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        feature_extractor=feature_extractor,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset is not None else None,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    if push_to_hub:
        repo_id = training_args.hub_model_id or os.path.basename(os.path.normpath(training_args.output_dir))
        logger.info("Pushing trained checkpoint to the Hub: %s", repo_id)
        model.push_to_hub(
            repo_id,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            private=bool(training_args.hub_private_repo),
            token=training_args.hub_token,
            revision=training_args.hub_revision,
        )


if __name__ == "__main__":
    sys.exit(main())
