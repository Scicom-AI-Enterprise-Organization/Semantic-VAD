"""Inference latency benchmark across variable audio input lengths, for the
Whisper-encoder -> adapter -> Qwen3 model (`semantic_vad/training/modeling.py::WhisperQwen3`)
-- the "branch we build" replacing the Qwen2-Audio classifier in `semvad/modeling.py`. Same
methodology as `scripts/benchmark_latency.py` so the two JSON outputs are directly comparable.

Measures wall-clock latency of a single **serving-style probe**: build the chat prompt up to
(not including) the assistant marker token, run one forward pass, and read the softmax over
{`<|eot|>`, `<|hold|>`} logits at the last position (INTEGRATION.md "EOT via logprobs" --
no classification head).

Note: `WhisperFeatureExtractor` also pads every clip to a fixed 30s mel grid, but unlike
Qwen2-Audio's processor, `feature_attention_mask` here tracks the *real* valid-frame count, so
the number of `<|AUDIO|>` placeholders (`num_audio_tokens`) scales with actual clip length --
the audio encoder's cost is ~flat (fixed 3000-frame input either way) but the Qwen3 trunk's
attention over the audio-token span does grow with duration, same as the Qwen2-Audio path.

Usage:
  python scripts/benchmark_latency_qwen3.py --device cuda
  python scripts/benchmark_latency_qwen3.py --device cuda --checkpoint runs/eot-qwen3 \\
      --durations 0.5 1 2 4 8 16 24 30 --repeats 20 --output latency_qwen3.json

On the GPU training pod (flash-attn installed), pass `--attn_implementation flash_attention_2`
to match the real training/serving kernel; the default `sdpa` runs anywhere (including a
laptop CPU) for a quick sanity check.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch


@dataclasses.dataclass
class LatencyStats:
    duration_s: float
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def to_row(self) -> str:
        return (
            f"{self.duration_s:>8.2f}  {self.n:>4d}  {self.mean_ms:>9.1f}  "
            f"{self.median_ms:>9.1f}  {self.p95_ms:>9.1f}  {self.min_ms:>9.1f}  "
            f"{self.max_ms:>9.1f}  {self.std_ms:>8.1f}"
        )


def summarize(duration_s: float, samples_ms: list) -> LatencyStats:
    arr = np.array(samples_ms)
    return LatencyStats(
        duration_s=duration_s,
        n=len(arr),
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
    )


def make_audio(duration_s: float, sampling_rate: int, seed: int) -> np.ndarray:
    """Synthetic clip -- forward-pass latency depends on shape, not content."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * sampling_rate)) * 0.05).astype(np.float32)


def build_probe_ids(encode, ids, *, n_audio: int, system: str) -> list:
    """Prompt up to (not including) the marker token -- the single-probe serving read.

    Mirrors `semantic_vad.training.prompt.build_example`'s system/user/assistant-header
    spans, stopping right before the marker so the forward pass scores it via logits
    instead of baking a label in.
    """
    input_ids: list = []

    def add(tokens: list) -> None:
        input_ids.extend(tokens)

    add([ids.im_start])
    add(encode("system\n"))
    add(encode(system))
    add([ids.im_end])
    add(encode("\n"))

    add([ids.im_start])
    add(encode("user\n"))
    add([ids.audio_bos])
    add([ids.audio] * n_audio)
    add([ids.audio_eos])
    add([ids.im_end])
    add(encode("\n"))

    add([ids.im_start])
    add(encode("assistant\n"))
    return input_ids


@torch.inference_mode()
def predict_p_eot(model, tokenizer, feature_extractor, special_ids, marker_ids, audio, sampling_rate, device, system):
    """Score one causal audio prefix as p(eot) via a single forward pass + marker softmax."""
    from semantic_vad.training.prompt import num_audio_tokens

    target_sr = feature_extractor.sampling_rate
    if sampling_rate != target_sr:
        import librosa

        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sampling_rate, target_sr=target_sr)

    feat = feature_extractor(
        audio, sampling_rate=target_sr, return_attention_mask=True, padding="max_length", return_tensors="np"
    )
    input_features = torch.from_numpy(feat["input_features"][0].astype(np.float32))[None].to(device)
    feature_attention_mask = torch.from_numpy(np.asarray(feat["attention_mask"][0], dtype=np.int64))[None].to(device)
    n_audio = num_audio_tokens(int(feature_attention_mask.sum().item()))

    input_ids = build_probe_ids(
        lambda text: tokenizer.encode(text, add_special_tokens=False), special_ids, n_audio=n_audio, system=system
    )
    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)[None]

    outputs = model(
        input_ids=input_ids_t,
        input_features=input_features.to(model.dtype),
        feature_attention_mask=feature_attention_mask,
    )
    logits = outputs.logits[0, -1]
    pair = torch.stack([logits[marker_ids["eot"]], logits[marker_ids["hold"]]])
    return torch.softmax(pair.float(), dim=0)[0].item()


def bench_one(
    model, tokenizer, feature_extractor, special_ids, marker_ids, audio, sampling_rate, device, system, warmup, repeats
) -> list:
    for _ in range(warmup):
        predict_p_eot(model, tokenizer, feature_extractor, special_ids, marker_ids, audio, sampling_rate, device, system)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_p_eot(model, tokenizer, feature_extractor, special_ids, marker_ids, audio, sampling_rate, device, system)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
    return samples_ms


def build_fresh_model(qwen3_name, whisper_name, dtype, attn_implementation, tokenizer):
    from transformers import AutoConfig, WhisperFeatureExtractor

    from semantic_vad.training.modeling import WhisperQwen3
    from semantic_vad.training.prompt import EOT_TOKEN, HOLD_TOKEN, SPECIAL_TOKENS, SpecialIds

    tokenizer.add_tokens(SPECIAL_TOKENS)
    special_ids = SpecialIds(
        im_start=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        im_end=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        audio_bos=tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        audio_eos=tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        audio=tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
    )
    marker_ids = {
        "eot": tokenizer.convert_tokens_to_ids(EOT_TOKEN),
        "hold": tokenizer.convert_tokens_to_ids(HOLD_TOKEN),
    }

    config = AutoConfig.from_pretrained(qwen3_name)
    whisper_config = AutoConfig.from_pretrained(whisper_name)
    encoder_config = getattr(whisper_config, "encoder", None) or whisper_config
    config.audio_encoder_config = encoder_config.to_dict()
    config.audio_token_index = special_ids.audio
    config.eot_token_index = marker_ids["eot"]
    config.hold_token_index = marker_ids["hold"]
    config.use_cache = False

    model = WhisperQwen3.from_pretrained(
        qwen3_name, config=config, attn_implementation=attn_implementation, torch_dtype=dtype
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_name)
    return model, feature_extractor, special_ids, marker_ids


def load_checkpoint_model(checkpoint, dtype, attn_implementation, tokenizer):
    from transformers import WhisperFeatureExtractor

    from semantic_vad.training.modeling import WhisperQwen3
    from semantic_vad.training.prompt import SpecialIds

    model = WhisperQwen3.from_pretrained(checkpoint, attn_implementation=attn_implementation, torch_dtype=dtype)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    special_ids = SpecialIds(
        im_start=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        im_end=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        audio_bos=tokenizer.convert_tokens_to_ids("<|audio_bos|>"),
        audio_eos=tokenizer.convert_tokens_to_ids("<|audio_eos|>"),
        audio=tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
    )
    marker_ids = {
        "eot": model.config.eot_token_index,
        "hold": model.config.hold_token_index,
    }
    return model, feature_extractor, special_ids, marker_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen3-name", default="Qwen/Qwen3-0.6B", help="ignored if --checkpoint is given")
    parser.add_argument("--whisper-name", default="openai/whisper-base", help="ignored if --checkpoint is given")
    parser.add_argument("--checkpoint", default=None, help="dir written by semantic_vad.training.train (trainer.save_model)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--durations", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16, 24, 30],
        help="audio clip durations in seconds to benchmark",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per duration, to settle caches/cudnn")
    parser.add_argument("--repeats", type=int, default=10, help="timed calls per duration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="optional path to dump results as JSON")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from semantic_vad.training.prompt import DEFAULT_SYSTEM

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tokenizer_source = args.checkpoint or args.qwen3_name
    print(f"[bench] loading tokenizer from {tokenizer_source} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if args.checkpoint:
        print(f"[bench] loading checkpoint {args.checkpoint} ({args.dtype}) on {args.device} ...")
        model, feature_extractor, special_ids, marker_ids = load_checkpoint_model(
            args.checkpoint, dtype, args.attn_implementation, tokenizer
        )
    else:
        print(
            f"[bench] building fresh {args.qwen3_name} + {args.whisper_name} "
            f"({args.dtype}) on {args.device} ..."
        )
        model, feature_extractor, special_ids, marker_ids = build_fresh_model(
            args.qwen3_name, args.whisper_name, dtype, args.attn_implementation, tokenizer
        )
    model.to(args.device)
    model.eval()

    header = (
        f"{'dur(s)':>8}  {'n':>4}  {'mean(ms)':>9}  {'p50(ms)':>9}  "
        f"{'p95(ms)':>9}  {'min(ms)':>9}  {'max(ms)':>9}  {'std(ms)':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for duration_s in args.durations:
        audio = make_audio(duration_s, args.sampling_rate, seed=args.seed)
        samples_ms = bench_one(
            model, tokenizer, feature_extractor, special_ids, marker_ids, audio, args.sampling_rate,
            args.device, DEFAULT_SYSTEM, args.warmup, args.repeats,
        )
        stats = summarize(duration_s, samples_ms)
        print(stats.to_row())
        results.append(dataclasses.asdict(stats))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
