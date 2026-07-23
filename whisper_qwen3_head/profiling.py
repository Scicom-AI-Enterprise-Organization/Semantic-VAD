"""`torch.profiler`-based `TrainerCallback` for `whisper_qwen3_head.train`, to find CPU vs.
GPU bottlenecks (dataloading/collation stalls vs. Whisper-encoder/Qwen3-backbone compute vs.
H2D copies) in a real training loop instead of guessing from wall-clock alone.

Captures a handful of steps mid-run (`wait` idle steps to let the CUDA allocator/cudnn
autotune settle, then `warmup` steps to prime the profiler itself, then `active` steps
actually recorded -- `torch.profiler.schedule`'s standard pattern) and writes a Chrome
trace + TensorBoard-plugin trace via `tensorboard_trace_handler`. By default it also stops
training right after capturing, since a profiling run is a dedicated short run, not a full
training job.

Usage (wired into `whisper_qwen3_head.train` via `--profile`)::

    torchrun --nproc_per_node=1 -m whisper_qwen3_head.train \\
        --output_dir /tmp/eot-profile --dataset_name en --streaming --max_steps 20 \\
        --per_device_train_batch_size 8 --report_to none \\
        --profile --profile_output_dir ./prof_logs

Inspect the result either as a quick console summary (printed automatically once the
capture finishes) or a full trace::

    tensorboard --logdir ./prof_logs   # needs `pip install torch-tb-profiler`
    # or open ./prof_logs/*.pt.trace.json in chrome://tracing or https://ui.perfetto.dev
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

import torch.profiler as tp
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProfilingArguments:
    profile: bool = dataclasses.field(
        default=False, metadata={"help": "capture a torch.profiler trace early in training, then stop"}
    )
    profile_output_dir: str = dataclasses.field(
        default="./prof_logs", metadata={"help": "dir for the TensorBoard-plugin + chrome trace files"}
    )
    profile_skip_first: int = dataclasses.field(
        default=0, metadata={"help": "steps to run untraced before the wait/warmup/active schedule starts"}
    )
    profile_wait: int = dataclasses.field(
        default=1, metadata={"help": "idle (untraced) steps per repeat, after profile_skip_first"}
    )
    profile_warmup: int = dataclasses.field(
        default=1, metadata={"help": "traced-but-discarded steps per repeat, to prime the profiler/CUDA caching"}
    )
    profile_active: int = dataclasses.field(default=3, metadata={"help": "recorded steps per repeat"})
    profile_repeat: int = dataclasses.field(
        default=1, metadata={"help": "how many wait/warmup/active cycles to capture"}
    )
    profile_record_shapes: bool = dataclasses.field(default=True, metadata={"help": "record op input tensor shapes"})
    profile_memory: bool = dataclasses.field(
        default=True, metadata={"help": "track tensor allocations/frees, for GPU/CPU memory bottlenecks"}
    )
    profile_with_stack: bool = dataclasses.field(
        default=True, metadata={"help": "record Python call stacks per op (needed for the TensorBoard source view)"}
    )
    profile_rank0_only: bool = dataclasses.field(
        default=True,
        metadata={"help": "only trace on the main process in a distributed run (one report is usually enough)"},
    )
    profile_stop_after: bool = dataclasses.field(
        default=True, metadata={"help": "stop training once the trace capture finishes (a profiling run is short)"}
    )

    def total_steps(self) -> int:
        """Steps `ProfilerCallback` needs to run through skip+wait+warmup+active x repeat."""
        return self.profile_skip_first + (self.profile_wait + self.profile_warmup + self.profile_active) * self.profile_repeat


class ProfilerCallback(TrainerCallback):
    """Drives a `torch.profiler.profile` context across `Trainer`'s step loop.

    `on_step_end` fires once per *global* optimizer step (after gradient accumulation), so
    with `--gradient_accumulation_steps > 1` each captured "step" in the trace actually spans
    several forward/backward micro-batches -- still fine for spotting a CPU-bound dataloader
    (gaps between GPU kernels) vs. a GPU-bound backbone (back-to-back CUDA kernels, little CPU
    idle), just coarser-grained than per-micro-batch.
    """

    def __init__(self, profiling_args: ProfilingArguments) -> None:
        self.args = profiling_args
        self.prof: Optional[tp.profile] = None
        self.step_count = 0
        self._active = True
        self._done = False

    def _is_main_process(self, args) -> bool:
        return not self.args.profile_rank0_only or args.process_index == 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._active = self._is_main_process(args)
        if not self._active:
            return
        os.makedirs(self.args.profile_output_dir, exist_ok=True)
        logger.info(
            "profiler: capturing %d steps (skip_first=%d wait=%d warmup=%d active=%d repeat=%d) -> %s",
            self.args.total_steps(),
            self.args.profile_skip_first,
            self.args.profile_wait,
            self.args.profile_warmup,
            self.args.profile_active,
            self.args.profile_repeat,
            self.args.profile_output_dir,
        )
        self.prof = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
            schedule=tp.schedule(
                skip_first=self.args.profile_skip_first,
                wait=self.args.profile_wait,
                warmup=self.args.profile_warmup,
                active=self.args.profile_active,
                repeat=self.args.profile_repeat,
            ),
            on_trace_ready=tp.tensorboard_trace_handler(self.args.profile_output_dir),
            record_shapes=self.args.profile_record_shapes,
            profile_memory=self.args.profile_memory,
            with_stack=self.args.profile_with_stack,
        )
        self.prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if not self._active or self.prof is None or self._done:
            return
        self.prof.step()
        self.step_count += 1
        if self.step_count >= self.args.total_steps():
            self._done = True
            self._print_summary()
            if self.args.profile_stop_after:
                control.should_training_stop = True

    def _print_summary(self) -> None:
        """Quick CPU/GPU bottleneck read straight in the console -- sorted by
        self time (time spent in the op itself, excluding children) so the actual
        hotspots surface instead of outer wrapper ops."""
        try:
            averages = self.prof.key_averages()
            print("\n[profiler] top ops by self CUDA time:")
            print(averages.table(sort_by="self_cuda_time_total", row_limit=15))
            print("\n[profiler] top ops by self CPU time:")
            print(averages.table(sort_by="self_cpu_time_total", row_limit=15))
        except Exception:
            logger.exception("profiler: failed to print summary table (trace files were still written)")

    def on_train_end(self, args, state, control, **kwargs):
        if not self._active or self.prof is None:
            return
        self.prof.__exit__(None, None, None)
        logger.info("profiler: trace written to %s (open with `tensorboard --logdir` or chrome://tracing)", self.args.profile_output_dir)
