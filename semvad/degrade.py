"""Call-centre / telephony channel degradation for training audio.

Ported from Sidon's `Degrader._telephony` (runpod/degradations.py) and trimmed to
just the telephony path -- the deployment target for this model is call-centre
audio, not the generic reverb/noise-corpus augmentations the source file also
supports, so those (and their `pyroomacoustics`/noise-filelist dependencies)
are deliberately left out here.

Chain: telephone high-pass -> a random narrowband ceiling (<~6 kHz, drawn per
call) realized via an 8/11/12/16 kHz bottleneck + lowpass -> a low-bitrate codec
round-trip (GSM = muffled ~1 kHz edge, mu-law/none = brighter ~3.4 kHz -- mixing
across calls brackets the measured spread of real recordings) -> a 16-40 kbps
MP3 container -> light line noise -> resampled back to the input rate.

Applied once per *turn* (call), not per truncated example -- a real phone
channel's distortion profile doesn't change mid-call, so every causal example
cut from the same turn should share one degraded realization. See
`iter_causal_examples` in `semvad/data.py` for the call site.

Codec round-trips, unlike the torchaudio.io-based original: `torchaudio.io`
(AudioEffector/CodecConfig) was removed outright in torchaudio>=2.9 (moved into
"maintenance mode", functionality migrated to torchcodec). torchcodec's own
encoder API (`torchcodec.encoders.AudioEncoder`) only takes a container `format`
with no encoder-name override, so there's no way to ask it for `pcm_mulaw`
inside a wav the way the old `encoder="pcm_mulaw"` kwarg could, and GSM isn't
present as an encodable codec in torchcodec's bundled ffmpeg at all (only as a
decoder) -- confirmed empirically, not assumed. So each codec here uses whatever
actually works:
  - mp3:   torchcodec's AudioEncoder/AudioDecoder (self-describing container,
           round-trips cleanly through the Python API -- verified).
  - mulaw: G.711 mu-law is a closed-form compand/expand formula, computed
           directly in numpy -- no codec round-trip needed at all, and no
           dependency on any encoder exposing it explicitly.
  - gsm:   shells out to the system `ffmpeg` binary via subprocess, since
           that's the only remaining avenue with real GSM 06.10 encode support
           (whether it's actually available depends on the ffmpeg build on the
           box -- probed once per process and dropped from the codec mix,
           weights renormalized, if it isn't).

Any transform that raises returns the input unchanged -- a bad sample must
never kill a training run.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from typing import Optional, Sequence

import numpy as np
import torch

try:
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder

    _HAS_TORCHCODEC = True
except Exception:  # noqa: BLE001
    _HAS_TORCHCODEC = False

try:
    from scipy.signal import butter, sosfilt

    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False

_GSM_ENCODE_AVAILABLE: Optional[bool] = None  # process-wide cache, probed lazily


def _probe_gsm_encode() -> bool:
    """One-time check for whether the system `ffmpeg` binary can actually
    *encode* GSM -- many builds (including torchcodec's bundled ffmpeg) support
    decoding it but not encoding it. Cached process-wide since this shells out."""
    global _GSM_ENCODE_AVAILABLE
    if _GSM_ENCODE_AVAILABLE is not None:
        return _GSM_ENCODE_AVAILABLE
    _GSM_ENCODE_AVAILABLE = False
    if shutil.which("ffmpeg"):
        silence = np.zeros(800, dtype=np.float32).tobytes()  # 0.1s @ 8kHz
        try:
            proc = subprocess.run(
                ["ffmpeg", "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "-", "-f", "gsm", "-y", "-"],
                input=silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            )
            _GSM_ENCODE_AVAILABLE = proc.returncode == 0 and len(proc.stdout) > 0
        except Exception:  # noqa: BLE001
            _GSM_ENCODE_AVAILABLE = False
    return _GSM_ENCODE_AVAILABLE


def _gsm_roundtrip(x: np.ndarray, sr: int) -> np.ndarray:
    """Encode/decode through raw GSM 06.10 (8kHz mono only) via the system
    ffmpeg binary. Returns `x` unchanged on any failure."""
    raw = np.ascontiguousarray(x, dtype=np.float32).tobytes()
    try:
        enc = subprocess.run(
            ["ffmpeg", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-f", "gsm", "-y", "-"],
            input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        if enc.returncode != 0 or not enc.stdout:
            return x
        dec = subprocess.run(
            ["ffmpeg", "-f", "gsm", "-i", "-", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-y", "-"],
            input=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        if dec.returncode != 0 or not dec.stdout:
            return x
        out = np.frombuffer(dec.stdout, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return x
    if len(out) >= len(x):
        return out[: len(x)].copy()
    return np.pad(out, (0, len(x) - len(out)))


def _mulaw_roundtrip(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 mu-law compand -> 8-bit quantize -> expand, in closed form."""
    x = np.clip(x, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    compressed = np.sign(x) * magnitude
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)
    compressed_q = codes / 255.0 * 2.0 - 1.0
    magnitude_inv = np.expm1(np.abs(compressed_q) * np.log1p(mu)) / mu
    return (np.sign(compressed_q) * magnitude_inv).astype(np.float32)


def _mp3_roundtrip(x: np.ndarray, sr: int, bit_rate: int) -> np.ndarray:
    if not _HAS_TORCHCODEC:
        return x
    try:
        samples = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)  # (1, N)
        encoded = AudioEncoder(samples, sample_rate=sr).to_tensor(format="mp3", bit_rate=bit_rate)
        decoded = AudioDecoder(encoded).get_all_samples()
        out = decoded.data[0].numpy().astype(np.float32)
    except Exception:  # noqa: BLE001
        return x
    if len(out) >= len(x):
        return out[: len(x)]
    return np.pad(out, (0, len(x) - len(out)))


class TelephonyDegrader:
    """Randomly degrades a mono waveform to sound like a call-centre recording.

    Construct once (e.g. per dataloader worker) and reuse. Call
    `degrade(audio, sampling_rate)`.
    """

    def __init__(
        self,
        apply_prob: float = 0.7,
        hp_hz: Sequence[float] = (200.0, 350.0),
        band_hz: Sequence[float] = (2800.0, 4200.0),
        codecs: Sequence[str] = ("gsm", "mulaw", "none"),
        codec_weights: Sequence[float] = (0.6, 0.25, 0.15),
        mp3_kbps: Sequence[int] = (16, 24, 32, 40),
        snr_db: Sequence[float] = (8.0, 28.0),
        noise_prob: float = 0.85,
        sr_choices: Sequence[int] = (8000, 11025, 12000, 16000),
        packet_loss_prob: float = 0.0,
    ):
        self.apply_prob = float(apply_prob)
        self.hp_hz = tuple(hp_hz)
        self.band_hz = tuple(band_hz)
        self.codecs = tuple(codecs)
        self.codec_weights = tuple(codec_weights)
        self.mp3_kbps = tuple(mp3_kbps)
        self.snr_db = tuple(snr_db)
        self.noise_prob = float(noise_prob)
        self.sr_choices = tuple(sorted(sr_choices))
        self.packet_loss_prob = float(packet_loss_prob)
        # mp3 is unconditional in `_telephony` below, so no torchcodec == no chain.
        if self.apply_prob > 0 and not _HAS_TORCHCODEC:
            print("[degrade] torchcodec encoders/decoders unavailable -- telephony degradation disabled")
            self.apply_prob = 0.0
        if not _HAS_SCIPY:
            print("[degrade] scipy unavailable -- telephone high-pass/low-pass filters disabled")
        if self.apply_prob > 0 and "gsm" in self.codecs and not _probe_gsm_encode():
            print("[degrade] system ffmpeg can't encode gsm -- dropping it from the codec mix")

    # -- filters ----------------------------------------------------------------
    def _highpass(self, x: np.ndarray, sr: int, fc: float) -> np.ndarray:
        if not _HAS_SCIPY or fc <= 0:
            return x
        sos = butter(2, min(fc, sr / 2 - 1) / (sr / 2), "highpass", output="sos")
        return sosfilt(sos, x).astype(np.float32)

    def _lowpass(self, x: np.ndarray, sr: int, fc: float) -> np.ndarray:
        if not _HAS_SCIPY or fc <= 0 or fc >= sr / 2:
            return x
        sos = butter(4, fc / (sr / 2), "lowpass", output="sos")
        return sosfilt(sos, x).astype(np.float32)

    def _pick_bottleneck(self, band: float) -> int:
        """Smallest valid sample rate whose Nyquist covers `band`."""
        for s in self.sr_choices:
            if s / 2.0 >= band:
                return int(s)
        return int(self.sr_choices[-1])

    def _codecs_for(self, ts: int):
        """GSM is 8 kHz-only, and needs a real encoder-capable ffmpeg -- drop it
        (renormalizing weights) whenever either condition isn't met."""
        gsm_ok = ts == 8000 and _probe_gsm_encode()
        cs, ws = [], []
        for c, w in zip(self.codecs, self.codec_weights):
            if c == "gsm" and not gsm_ok:
                continue
            cs.append(c)
            ws.append(w)
        return (cs, ws) if cs else (["none"], [1.0])

    def _mix_noise(self, x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        sig_p = float(np.mean(x**2)) + 1e-12
        noi_p = float(np.mean(noise**2)) + 1e-12
        scale = (sig_p / (noi_p * (10 ** (snr_db / 10.0)))) ** 0.5
        return (x + scale * noise).astype(np.float32)

    def _packet_loss(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Zero out a few short random chunks -- VoIP dropouts."""
        total = len(x) / sr
        n_chunks = int(total * 3 / 10)
        x = x.copy()
        for _ in range(n_chunks):
            dur = random.uniform(0.02, 0.2)
            if total - dur <= 0:
                break
            start = random.uniform(0, total - dur)
            i0, i1 = int(start * sr), int((start + dur) * sr)
            x[i0:i1] = 0.0
        return x

    # -- chain -----------------------------------------------------------------
    def _telephony(self, x: np.ndarray, sr: int, trace: Optional[dict] = None) -> np.ndarray:
        import librosa

        hp = random.uniform(*self.hp_hz)
        y = self._highpass(x, sr, hp)
        band = random.uniform(*self.band_hz)  # random narrowband ceiling (< ~6 kHz)
        ts = self._pick_bottleneck(band)
        y = librosa.resample(y, orig_sr=sr, target_sr=ts).astype(np.float32) if sr != ts else y.copy()
        y = self._lowpass(y, ts, band)  # realize the random ceiling
        peak = float(np.abs(y).max())
        if peak > 1e-6:
            y = y / peak
        cs, ws = self._codecs_for(ts)
        codec = random.choices(cs, weights=ws, k=1)[0]
        if codec == "gsm":
            y = _gsm_roundtrip(y, ts)
        elif codec == "mulaw":
            y = _mulaw_roundtrip(y)
        # final low-bitrate MP3 container
        br = int(random.choice(self.mp3_kbps)) * 1000
        y = _mp3_roundtrip(y, ts, br)
        noised = random.random() < self.noise_prob
        snr = random.uniform(*self.snr_db) if noised else None
        if noised:
            y = self._mix_noise(y, np.random.randn(len(y)).astype(np.float32), snr)
        if trace is not None:
            trace.update(hp_hz=hp, band_hz=band, bottleneck_sr=ts, codec=codec, mp3_bps=br, snr_db=snr)
        return librosa.resample(y, orig_sr=ts, target_sr=sr).astype(np.float32) if sr != ts else y

    def degrade(self, audio: np.ndarray, sampling_rate: int, trace: Optional[dict] = None) -> np.ndarray:
        """With probability `apply_prob`, run the telephony chain; otherwise
        return `audio` unchanged. Always returns mono float32, same length,
        clipped to [-1, 1]. If `trace` (a dict) is given, it's filled in with the
        parameters actually chosen for this call (codec, band, bitrate, SNR, ...)
        -- handy for inspecting/previewing what the augmentation is doing."""
        x = np.asarray(audio, dtype=np.float32)
        if trace is not None:
            trace.clear()
            trace["applied"] = False
        if self.apply_prob <= 0 or random.random() >= self.apply_prob:
            return x
        try:
            x = self._telephony(x, sampling_rate, trace=trace)
            if self.packet_loss_prob and random.random() < self.packet_loss_prob:
                x = self._packet_loss(x, sampling_rate)
                if trace is not None:
                    trace["packet_loss"] = True
            if trace is not None:
                trace["applied"] = True
        except Exception as e:  # noqa: BLE001 -- never let augmentation kill a sample
            print(f"[degrade] telephony chain failed, using clean input: {e}")
            return np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(x, -1.0, 1.0).astype(np.float32)
