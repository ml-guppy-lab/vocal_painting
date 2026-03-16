"""
vocal.py — Phase 2: Live Voice Feature Extraction
--------------------------------------------------
Streams mic input in real time and prints pitch, amplitude,
and spectral centroid every ~185 ms using a sounddevice callback.

All logic is in standalone functions so individual features
can be imported and reused in future modules.
"""

import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import librosa

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # Hz — librosa's native rate
FRAME_SAMPLES  = 4096    # ~185 ms per analysis frame
CALLBACK_BLOCK = 1024    # sounddevice internal block size (lower = less latency)
FMIN           = 50      # Hz — lowest expected vocal pitch
FMAX           = 1000    # Hz — highest expected vocal pitch
UNVOICED_THRESH = 900    # YIN returns ~FMAX for unvoiced frames; filter these out


# ── Feature extractors ────────────────────────────────────────────────────────

def extract_pitch(frame: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate fundamental frequency (Hz) via the YIN algorithm.
    Returns NaN for unvoiced / silent frames.
    """
    f0        = librosa.yin(frame, fmin=FMIN, fmax=FMAX, sr=sr)
    f0_voiced = f0[f0 < UNVOICED_THRESH]
    return float(np.median(f0_voiced)) if len(f0_voiced) > 0 else float("nan")


def extract_amplitude(frame: np.ndarray) -> float:
    """Mean RMS energy of the frame (proxy for loudness)."""
    rms = librosa.feature.rms(y=frame)
    return float(np.mean(rms))


def extract_spectral_centroid(frame: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Mean spectral centroid (Hz) — higher = brighter / breathier tone."""
    centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
    return float(np.mean(centroid))


def extract_features(frame: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Bundle all three features for one audio frame.

    Returns
    -------
    {
        "pitch_hz"            : float  (NaN if unvoiced),
        "amplitude_rms"       : float,
        "spectral_centroid_hz": float,
    }
    """
    return {
        "pitch_hz":             extract_pitch(frame, sr),
        "amplitude_rms":        extract_amplitude(frame),
        "spectral_centroid_hz": extract_spectral_centroid(frame, sr),
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_features(features: dict) -> None:
    """Print a feature dict as a single console line."""
    pitch = features["pitch_hz"]
    amp   = features["amplitude_rms"]
    cent  = features["spectral_centroid_hz"]

    pitch_str = f"{pitch:7.1f} Hz" if not np.isnan(pitch) else "  ---    "
    print(
        f"  Pitch: {pitch_str}  |  "
        f"Amplitude: {amp:.5f}  |  "
        f"Centroid: {cent:7.1f} Hz",
        flush=True,
    )


# ── Live stream loop ──────────────────────────────────────────────────────────

def live_feature_loop(
    duration: Optional[float] = None,
    sr: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
) -> None:
    """
    Open a live mic stream and print extracted features for every frame.

    Parameters
    ----------
    duration      : run for this many seconds then stop (None = run until Ctrl+C)
    sr            : sample rate in Hz
    frame_samples : samples per analysis frame (~185 ms at 22050 Hz)
    """
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[stream] {status}", flush=True)
        audio_q.put(indata[:, 0].copy())   # mono slice

    frame_ms = frame_samples / sr * 1000
    print("=" * 68)
    print("  Vocal Painter — Phase 2: Live Feature Extraction")
    print("=" * 68)
    print(f"  Sample rate : {sr} Hz  |  Frame : {frame_samples} samples (~{frame_ms:.0f} ms)")
    if duration:
        print(f"  Running for {duration} s — press Ctrl+C to stop early.")
    else:
        print("  Hum or sing — press Ctrl+C to stop.")
    print()
    header = f"  {'Pitch':>12}      {'Amplitude':>12}     {'Centroid':>12}"
    print(header)
    print("-" * len(header))

    buffer = np.zeros(0, dtype=np.float32)
    start  = time.time()

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=CALLBACK_BLOCK,
        callback=_callback,
    ):
        try:
            while True:
                if duration and (time.time() - start) >= duration:
                    break
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])

                while len(buffer) >= frame_samples:
                    frame  = buffer[:frame_samples]
                    buffer = buffer[frame_samples:]
                    print_features(extract_features(frame, sr))

        except KeyboardInterrupt:
            print("\n  Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    live_feature_loop()
