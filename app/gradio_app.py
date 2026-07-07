"""Realtime p(EoT) demo -- mirrors how LiveKit's turn-detector playground shows a
live probability trace as you speak and pause (see the "Evaluation Model" /
policy-sweep description in README.md).

Two backends:
  --mock             heuristic, trailing-silence-duration predictor. No model
                      download, no GPU -- for exercising the streaming/UI
                      plumbing on modest hardware.
  (default)           the real Qwen2-Audio classification-head model from
                      semvad.modeling. This is a 7B-parameter audio-language
                      model; a 16GB unified-memory laptop will not run it
                      comfortably. Use --mock locally and point --device at a
                      real GPU box for the genuine model.

Usage:
  python app/gradio_app.py --mock
  python app/gradio_app.py --checkpoint runs/eot-v1 --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import deque
from typing import Optional

import numpy as np

MAX_WINDOW_SECONDS = 16.0  # causal audio context kept for scoring (bounded, per README §6)
HISTORY_SECONDS = 30.0  # length of the p(eot) trace shown on the plot
SILENCE_RMS_DBFS = -40.0  # energy threshold below which a frame counts as "silence"


def rms_dbfs(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))) + 1e-9)
    return 20.0 * np.log10(rms + 1e-9)


def trailing_silence_seconds(audio: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Seconds of trailing audio below the silence energy threshold."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_len
    silence = 0.0
    for i in range(n_frames - 1, -1, -1):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        if rms_dbfs(frame) > SILENCE_RMS_DBFS:
            break
        silence += frame_ms / 1000
    return silence


class MockPredictor:
    """No model download -- p(eot) rises with trailing silence duration."""

    def predict_p_eot(self, audio: np.ndarray, sampling_rate: int, prior_text: str = "") -> float:
        silence = trailing_silence_seconds(audio, sampling_rate)
        return float(1.0 - np.exp(-silence / 0.6))


class RealPredictor:
    """Wraps the fine-tuned (or base, untrained-head) Qwen2AudioEoTClassifier."""

    def __init__(self, model_name: str, checkpoint: Optional[str], device: str):
        import torch
        from transformers import AutoProcessor

        from semvad.modeling import Qwen2AudioEoTClassifier

        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = Qwen2AudioEoTClassifier.from_pretrained(model_name, dtype=dtype)
        if checkpoint:
            self.model.load_adapter(checkpoint)
        else:
            print(
                "[semvad] No --checkpoint given: the EoT head has random weights, "
                "so p(eot) will be meaningless until you load a fine-tuned adapter."
            )
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "right"

    def predict_p_eot(self, audio: np.ndarray, sampling_rate: int, prior_text: str = "") -> float:
        return self.model.predict_p_eot(self.processor, audio, sampling_rate, prior_text=prior_text)


@dataclasses.dataclass
class TurnPolicy:
    """Mirrors eot-bench's three swept knobs: `threshold`, `action_delay`, `timeout`."""

    threshold: float = 0.5
    action_delay: float = 0.2  # min silence duration before acting on the score
    timeout: float = 3.0  # max silence duration before ending the turn regardless

    def decide(self, p_eot: float, silence_seconds: float) -> str:
        if silence_seconds <= 0:
            return "listening"
        if silence_seconds >= self.timeout:
            return "end_of_turn (timeout)"
        if silence_seconds >= self.action_delay and p_eot >= self.threshold:
            return "end_of_turn"
        return "holding"


STATUS_COLOR = {
    "listening": "#2563eb",
    "holding": "#d97706",
    "end_of_turn": "#16a34a",
    "end_of_turn (timeout)": "#16a34a",
}


@dataclasses.dataclass
class SessionState:
    buffer: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sr: int = 16000
    t0: float = dataclasses.field(default_factory=time.time)
    history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=2000))  # (t, p_eot)
    last_p_eot: float = 0.0

    def append(self, sr: int, chunk: np.ndarray, max_seconds: float = MAX_WINDOW_SECONDS) -> None:
        self.sr = sr
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)
        # int16-range PCM comes back as large integers from some mic backends;
        # normalize defensively so the RMS/dBFS math stays sane either way.
        if chunk.size and np.max(np.abs(chunk)) > 4.0:
            chunk = chunk / 32768.0
        self.buffer = np.concatenate([self.buffer, chunk])
        max_len = int(max_seconds * sr)
        if len(self.buffer) > max_len:
            self.buffer = self.buffer[-max_len:]


def render_plot(history: deque, threshold: float):
    import matplotlib.pyplot as plt

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 2.5))
    if history:
        ts, ps = zip(*history)
        ax.plot(ts, ps, color="#4f46e5", linewidth=2)
    ax.axhline(threshold, color="#ef4444", linestyle="--", linewidth=1, label="threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("p(eot)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def status_html(status: str, p_eot: float, silence: float) -> str:
    color = STATUS_COLOR.get(status, "#6b7280")
    return (
        f"<div style='font-size:1.4rem;font-weight:600;color:{color}'>{status.upper()}</div>"
        f"<div style='color:#6b7280'>p(eot)={p_eot:.3f} &nbsp;|&nbsp; silence={silence:.2f}s</div>"
    )


def build_demo(predictor, window_seconds: float = MAX_WINDOW_SECONDS):
    import gradio as gr

    def on_chunk(stream_audio, state, threshold, action_delay, timeout):
        if state is None:
            state = SessionState()
        if stream_audio is None:
            return render_plot(state.history, threshold), status_html("listening", 0.0, 0.0), state

        sr, chunk = stream_audio
        state.append(sr, chunk, max_seconds=window_seconds)

        silence = trailing_silence_seconds(state.buffer, state.sr)
        # TurnPolicy.decide() ignores p_eot entirely while silence == 0 (still "listening"),
        # so skip the 7B forward pass during active speech -- only score once a pause starts.
        if silence > 0:
            state.last_p_eot = predictor.predict_p_eot(state.buffer, state.sr)
        p_eot = state.last_p_eot
        status = TurnPolicy(threshold=threshold, action_delay=action_delay, timeout=timeout).decide(p_eot, silence)

        t = time.time() - state.t0
        state.history.append((t, p_eot))
        while len(state.history) > 1 and t - state.history[0][0] > HISTORY_SECONDS:
            state.history.popleft()

        if status.startswith("end_of_turn"):
            # turn ended -- start listening fresh for the next one
            state.buffer = np.zeros(0, dtype=np.float32)
            state.last_p_eot = 0.0

        return render_plot(state.history, threshold), status_html(status, p_eot, silence), state

    def reset_state():
        return SessionState(), render_plot(deque(), 0.5), status_html("listening", 0.0, 0.0)

    with gr.Blocks(title="Semantic VAD -- live p(EoT)") as demo:
        gr.Markdown(
            "## Semantic VAD -- live end-of-turn probability\n"
            "Speak, then pause. `p(eot)` should stay low through mid-turn hesitations "
            "and rise once you're actually done talking -- the same causal decision "
            "`eot-bench` scores, just watched live instead of swept offline."
        )
        state = gr.State(SessionState())
        with gr.Row():
            with gr.Column(scale=1):
                mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Microphone")
                clear_btn = gr.Button("Reset session")
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="threshold")
                action_delay = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="action_delay (s)")
                timeout = gr.Slider(0.5, 6.0, value=3.0, step=0.1, label="timeout (s)")
            with gr.Column(scale=2):
                status = gr.HTML(status_html("listening", 0.0, 0.0))
                plot = gr.Plot(render_plot(deque(), 0.5))

        mic.stream(
            on_chunk,
            inputs=[mic, state, threshold, action_delay, timeout],
            outputs=[plot, status, state],
            stream_every=0.3,
        )
        clear_btn.click(reset_state, outputs=[state, plot, status])

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="use the heuristic predictor, no model download")
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="dir written by Qwen2AudioEoTClassifier.save_adapter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-seconds", type=float, default=MAX_WINDOW_SECONDS)
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="expose a public Gradio link (off by default)")
    args = parser.parse_args()

    predictor = MockPredictor() if args.mock else RealPredictor(args.model_name, args.checkpoint, args.device)
    demo = build_demo(predictor, window_seconds=args.window_seconds)
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
