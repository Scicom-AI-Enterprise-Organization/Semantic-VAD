"""End-to-end integration test for `semvad/train.py`'s Trainer wiring.

Uses a tiny random-weight backbone (same trick as `scripts/smoke_test.py`) but
keeps `vocab_size`/`audio_token_index` identical to the real
Qwen/Qwen2-Audio-7B-Instruct config, and keeps the audio encoder's
`num_mel_bins`/`max_source_positions` identical too -- both are dictated by the
*real* downloaded processor's WhisperFeatureExtractor, which this script reuses
as-is. That lets the real tokenizer/feature-extractor/`EoTCollator` run
unmodified against a backbone cheap enough to train on a laptop CPU in seconds,
so this actually exercises `HfArgumentParser` dataclass parsing, `Trainer`'s
`remove_unused_columns=False` path, `compute_loss` reading `EoTOutput` as a
dict, `EoTTrainer._save`'s adapter-only checkpointing, and the eval/metrics
loop -- not just "does the model forward-pass," which `scripts/smoke_test.py`
already covers.

No 7B weights are downloaded. Only the small processor config (already cached
after running the app/adapter smoke tests) is needed.
"""

import shutil
import sys
import tempfile

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoProcessor, Qwen2AudioConfig, Qwen2AudioEncoderConfig, Qwen2AudioModel, TrainingArguments
from transformers.models.qwen2 import Qwen2Config

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from semvad.data import EoTCollator, expand_to_causal_examples  # noqa: E402
from semvad.modeling import EoTHead, EoTHeadConfig, Qwen2AudioEoTClassifier  # noqa: E402
from semvad.train import EoTTrainer, ModelArguments, compute_metrics, preprocess_logits_for_metrics  # noqa: E402

REAL_MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"


def make_row(rid: str, spans: list[dict], dur: float, sr: int = 16000) -> dict:
    return {
        "id": rid,
        "audio": {"array": (np.random.randn(int(dur * sr)) * 0.05).astype(np.float32), "sampling_rate": sr},
        "language": "en",
        "duration": dur,
        "silence_spans": spans,
        "words": [],
        "messages": [{"role": "user", "content": "hello there"}],
    }


def build_tiny_but_real_vocab_config(real_config: Qwen2AudioConfig) -> Qwen2AudioConfig:
    audio_config = Qwen2AudioEncoderConfig(
        num_mel_bins=real_config.audio_config.num_mel_bins,  # tied to the real feature extractor, can't shrink
        max_source_positions=real_config.audio_config.max_source_positions,  # ditto
        d_model=32,
        encoder_attention_heads=2,
        encoder_ffn_dim=64,
        encoder_layers=1,
    )
    text_config = Qwen2Config(
        vocab_size=real_config.text_config.vocab_size,  # tied to the real tokenizer
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return Qwen2AudioConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_index=real_config.audio_token_index,  # tied to the real tokenizer's <|AUDIO|> id
    )


def main():
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(REAL_MODEL_NAME)
    real_config = Qwen2AudioConfig.from_pretrained(REAL_MODEL_NAME)
    tiny_config = build_tiny_but_real_vocab_config(real_config)

    backbone = Qwen2AudioModel(tiny_config)
    backbone.audio_tower.requires_grad_(False)
    head = EoTHead(tiny_config.text_config.hidden_size, EoTHeadConfig(head_hidden_size=16, dropout=0.0))
    model = Qwen2AudioEoTClassifier(backbone, head)
    model.apply_lora(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])

    n_total = model.count_parameters()
    n_trainable = model.count_parameters(trainable_only=True)
    print(f"tiny model: total={n_total:,} trainable={n_trainable:,}")

    rows = [
        make_row("a", [{"start": 0.6, "end": 0.9}], dur=0.9),
        make_row("b", [{"start": 0.3, "end": 0.45}, {"start": 0.7, "end": 1.0}], dur=1.0),
        make_row("c", [{"start": 0.5, "end": 0.8}], dur=0.8),
        make_row("d", [{"start": 0.4, "end": 0.5}, {"start": 0.8, "end": 1.1}], dur=1.1),
    ]
    ds = Dataset.from_list(rows)
    expanded = ds.map(expand_to_causal_examples, batched=True, remove_columns=ds.column_names)
    print(f"expanded {len(rows)} rows -> {len(expanded)} causal examples")
    assert len(expanded) > len(rows)

    collator = EoTCollator(processor, max_audio_seconds=4.0)

    tmp_dir = tempfile.mkdtemp(prefix="semvad_train_smoke_")
    try:
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=3,
            logging_steps=1,
            save_steps=3,
            eval_strategy="steps",
            eval_steps=3,
            report_to=[],
            use_cpu=True,
            remove_unused_columns=False,
            label_names=["labels"],
            learning_rate=1e-3,
        )

        trainer = EoTTrainer(
            model=model,
            args=training_args,
            train_dataset=expanded,
            eval_dataset=expanded,
            data_collator=collator,
            processing_class=processor,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        train_result = trainer.train()
        print("train_result.metrics:", train_result.metrics)
        assert np.isfinite(train_result.metrics["train_loss"])

        eval_metrics = trainer.evaluate()
        print("eval_metrics:", eval_metrics)
        assert "eval_accuracy" in eval_metrics

        trainer.save_model(tmp_dir)
        import os

        saved = set(os.listdir(tmp_dir))
        print("output_dir contents:", saved)
        assert "eot_head.pt" in saved, "adapter-only save did not write eot_head.pt"
        assert "lora" in saved, "adapter-only save did not write the LoRA dir"
        assert not any(f.startswith("model") and f.endswith(".safetensors") for f in saved), (
            "a full model safetensors file was written -- _save fell back to the default full-checkpoint path"
        )
        print("OK: adapter-only checkpoint confirmed (no full 7B-scale state dict on disk)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Separately: confirm HfArgumentParser parses our three dataclasses without
    # field collisions against TrainingArguments (a distinct failure mode from
    # the actual training loop above).
    from transformers import HfArgumentParser

    from semvad.train import DataArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    argv = [
        "--output_dir", "/tmp/unused",
        "--dataset_name", "en",
        "--max_steps", "1",
        "--per_device_train_batch_size", "1",
    ]
    parsed_model_args, parsed_data_args, parsed_training_args = parser.parse_args_into_dataclasses(argv)
    assert parsed_data_args.dataset_name == "en"
    assert parsed_training_args.max_steps == 1
    print("OK: HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)) parses without collisions")

    print("ALL TRAIN-SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
