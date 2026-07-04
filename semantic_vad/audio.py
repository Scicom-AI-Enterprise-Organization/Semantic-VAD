"""Audio slicing and :class:`Turn` -> :class:`EOTRow` assembly (with time re-zeroing)."""

from __future__ import annotations

import io

import numpy as np

from .schema import EOTRow, Turn


def encode_wav(array: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    """Encode a mono float32 array to WAV bytes.

    This is how the audio is stored in the parquet (as ``{"bytes", "path"}``), matching
    `livekit/eot-bench-data`: the harness decodes the bytes with soundfile. Doing the
    encoding ourselves avoids the torch/torchcodec dependency ``datasets`` otherwise needs
    to serialize an ``Audio`` array.
    """
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, to_mono_f32(array), int(sr), format="WAV", subtype=subtype)
    return buf.getvalue()


def to_mono_f32(array: np.ndarray) -> np.ndarray:
    """Coerce an audio array to mono float32 in ``[-1, 1]``."""
    a = np.asarray(array)
    if a.ndim == 2:  # (samples, channels) or (channels, samples)
        # datasets/soundfile give (samples, channels); average channels.
        ax = 1 if a.shape[0] >= a.shape[1] else 0
        a = a.mean(axis=ax)
    if a.dtype.kind in ("i", "u"):
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    return a.astype(np.float32, copy=False)


def resample_linear(array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio to ``target_sr`` with linear interpolation.

    Good enough for an EOT benchmark (we care about timing and coarse spectral cues, not
    hi-fi). Avoids a heavyweight scipy/librosa dependency.
    """
    a = to_mono_f32(array)
    if sr == target_sr or len(a) == 0:
        return a
    n_out = int(round(len(a) * target_sr / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(a))
    x_new = np.linspace(0, len(a) - 1, n_out)
    return np.interp(x_new, x_old, a).astype(np.float32)


def slice_window(array: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Cut ``[start, end)`` seconds out of ``array``; zero-pad if ``end`` runs past the clip.

    Padding is what lets a turn have a real trailing EOT silence even when the source
    recording stops right after the last word.
    """
    a = to_mono_f32(array)
    i0 = max(0, int(round(start * sr)))
    i1 = int(round(end * sr))
    n_want = max(0, i1 - i0)
    chunk = a[i0 : min(i1, len(a))]
    if len(chunk) < n_want:
        chunk = np.concatenate([chunk, np.zeros(n_want - len(chunk), dtype=np.float32)])
    return chunk


def turn_to_row(
    turn: Turn,
    array: np.ndarray,
    sr: int,
    *,
    row_id: str,
    language: str,
    messages: list[dict[str, str]] | None = None,
) -> EOTRow:
    """Slice the turn's audio window and re-zero all times to the clip start."""
    clip = slice_window(array, sr, turn.window_start, turn.window_end)
    offset = turn.window_start
    duration = len(clip) / sr

    words = [
        {"word": w.word, "start": round(w.start - offset, 3), "end": round(w.end - offset, 3)}
        for w in turn.words
    ]
    spans = [
        {"start": round(s.start - offset, 3), "end": round(s.end - offset, 3)}
        for s in turn.silence_spans
    ]
    return EOTRow(
        id=row_id,
        audio=clip,
        sampling_rate=sr,
        language=language,
        duration=duration,
        silence_spans=spans,
        words=words,
        messages=messages or [{"role": "user", "content": turn.transcript}],
    )
