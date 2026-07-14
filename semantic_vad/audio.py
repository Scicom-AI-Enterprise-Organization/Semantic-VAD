"""Audio slicing and :class:`Turn` -> :class:`EOTRow` assembly (with time re-zeroing)."""

from __future__ import annotations

import io

import numpy as np

from .schema import EOTRow, Turn


#: file extension per stored audio format.
AUDIO_EXT = {"wav": "wav", "flac": "flac", "mp3": "mp3", "ogg": "ogg"}


def encode_wav(array: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    """Encode a mono float32 array to WAV bytes (kept for the tests / default path)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, to_mono_f32(array), int(sr), format="WAV", subtype=subtype)
    return buf.getvalue()


def encode_audio(array: np.ndarray, sr: int, fmt: str = "wav", mp3_bitrate: int = 32,
                 mp3_quality: int = 7) -> bytes:
    """Encode a mono float32 array to ``fmt`` bytes for storage in the parquet.

    ``wav``/``flac``/``ogg`` go through soundfile; ``mp3`` uses ``lameenc`` (libsndfile can
    decode mp3 but not encode it). mp3 at ~48 kbps mono is tiny and ample for EOT cues
    (timing + coarse prosody), and matches the Malaysian source's native format. All formats
    decode back the eot-bench way: ``soundfile.read(BytesIO(bytes))``.
    """
    a = to_mono_f32(array)
    if fmt == "mp3":
        import lameenc

        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
        enc = lameenc.Encoder()
        enc.set_bit_rate(mp3_bitrate)
        enc.set_in_sample_rate(int(sr))
        enc.set_channels(1)
        enc.set_quality(mp3_quality)  # 0=best/slowest .. 9=fastest; 7 is fast + fine for speech
        out = enc.encode(pcm.tobytes())
        out += enc.flush()
        return bytes(out)

    import soundfile as sf

    buf = io.BytesIO()
    subtype = {"wav": "PCM_16", "flac": "PCM_16"}.get(fmt)
    sf.write(buf, a, int(sr), format=fmt.upper(), subtype=subtype)
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


def speech_end_index(clip: np.ndarray, sr: int, thresh_rel: float = 0.1,
                     frame_ms: int = 20, hop_ms: int = 10) -> int:
    """Sample index where speech energy ends (simple energy VAD).

    Returns the end of the last frame whose RMS exceeds ``thresh_rel`` * the clip's loud
    level. Used to end a turn's clip in the trailing *non-activity* period so it never ends
    on a (partial) word -- a proper end-of-turn clip should decay to near-silence.
    """
    n = len(clip)
    if n == 0:
        return 0
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    rms, ends = [], []
    j = 0
    while j + frame <= n:
        rms.append(float(np.sqrt(np.mean(clip[j:j + frame] ** 2))))
        ends.append(j + frame)
        j += hop
    if not rms:
        return n
    rms = np.asarray(rms)
    thresh = max(1e-3, thresh_rel * float(np.percentile(rms, 95)))
    active = np.nonzero(rms > thresh)[0]
    return int(ends[active[-1]]) if len(active) else 0


def turn_to_row(
    turn: Turn,
    array: np.ndarray,
    sr: int,
    *,
    row_id: str,
    language: str,
    messages: list[dict[str, str]] | None = None,
    trim_trailing: bool = True,
    trailing_pad: float = 0.15,
    max_tail_ratio: float = 0.4,
) -> EOTRow | None:
    """Slice the turn's audio window, VAD-trim the tail to silence, re-zero times to the clip.

    ``trim_trailing`` uses an energy VAD to find the true end of speech and keeps only
    ``trailing_pad`` seconds of the following silence, so the clip never ends on a partial
    word. A real end-of-turn must decay to silence, so if the tail (last 50 ms) is still loud
    relative to the clip (> ``max_tail_ratio`` * clip RMS) -- e.g. music, or the speaker
    continued with no pause -- the turn is **dropped** (returns ``None``). Set
    ``max_tail_ratio`` high (e.g. 1.0) to disable the filter.
    """
    clip = slice_window(array, sr, turn.window_start, turn.window_end)
    offset = turn.window_start

    if trim_trailing and len(clip):
        se = speech_end_index(clip, sr)
        end = min(len(clip), se + int(round(trailing_pad * sr)))
        last_word_idx = int(round((turn.words[-1].end - offset) * sr))
        end = max(end, min(len(clip), last_word_idx))  # never cut into the last word
        clip = clip[:end]
    duration = len(clip) / sr

    # Quality gate: a proper EOT clip ends in silence. Drop clips whose tail is still loud
    # (no real trailing silence exists -- music or an immediate continuation).
    clip_rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
    tail = clip[-int(round(0.05 * sr)):] if len(clip) else clip
    tail_rms = float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 0.0
    if clip_rms > 1e-3 and tail_rms > max(0.04, max_tail_ratio * clip_rms):
        return None

    words = [
        {"word": w.word, "start": round(w.start - offset, 3), "end": round(w.end - offset, 3)}
        for w in turn.words
    ]
    spans = []
    for s in turn.silence_spans:
        st = round(s.start - offset, 3)
        en = round(min(s.end - offset, duration), 3)
        if en > st:
            spans.append({"start": st, "end": en})
    if spans:  # the eot silence runs to the (trimmed) clip end
        spans[-1]["end"] = round(duration, 3)
    # A valid row must end with a real EOT silence span; drop degenerate ones (the clip
    # trimmed flush to the last word, so there's no trailing pause to label `eot`).
    if not spans or (spans[-1]["end"] - spans[-1]["start"]) < 0.08:
        return None
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
