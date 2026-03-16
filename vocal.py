import os
import time
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf   # installed automatically as a librosa dependency
import librosa

DURATION        = 10      # seconds to record
SAMPLE_RATE     = 22050   # Hz  (librosa's default)
CHANNELS        = 1
SEGMENT_SECS    = 2       # analyse every N seconds
FMIN, FMAX      = 50, 1000  # Hz range for YIN pitch detection
UNVOICED_THRESH = 900     # YIN returns ~fmax for unvoiced frames — skip those

# ── Recording ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Vocal Painter — Pitch / Amplitude / Spectral-Centroid Test")
print("=" * 60)
print()
print("Sing the following, holding each note steadily:")
print("  [0 – 3 s]  HIGH note")
print("  [3 – 6 s]  LOW  note")
print("  [6 – 8 s]  LOUD note")
print("  [8 – 10 s] SOFT note")
print()
print("Recording starts in 2 seconds …")
time.sleep(2)

print(">>> RECORDING — go! <<<")
raw = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
)
sd.wait()
print(">>> Recording finished.\n")

# ── Save → reload with librosa ────────────────────────────────────────────────
audio_mono = raw.flatten()
tmp_path   = os.path.join(tempfile.gettempdir(), "vocal_painter_test.wav")
sf.write(tmp_path, audio_mono, SAMPLE_RATE)
print(f"Saved to temp file: {tmp_path}")

y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
print(f"Loaded with librosa  |  samples: {len(y)}  |  sr: {sr} Hz  |  duration: {len(y)/sr:.2f} s\n")

# ── Segment-by-segment analysis ───────────────────────────────────────────────
n_segments     = DURATION // SEGMENT_SECS
segment_samples = int(SEGMENT_SECS * sr)

header = f"{'Segment':<10} {'Time range':<14} {'Pitch (Hz)':>12} {'Amplitude (RMS)':>17} {'Spec. Centroid (Hz)':>21}"
print(header)
print("-" * len(header))

for i in range(n_segments):
    start   = i * segment_samples
    end     = min(start + segment_samples, len(y))
    segment = y[start:end]

    # — Fundamental frequency (YIN algorithm)
    f0          = librosa.yin(segment, fmin=FMIN, fmax=FMAX, sr=sr)
    f0_voiced   = f0[f0 < UNVOICED_THRESH]
    pitch_hz    = float(np.median(f0_voiced)) if len(f0_voiced) > 0 else float("nan")

    # — Amplitude (root-mean-square energy)
    rms         = librosa.feature.rms(y=segment)
    amplitude   = float(np.mean(rms))

    # — Spectral centroid ("brightness")
    centroid    = librosa.feature.spectral_centroid(y=segment, sr=sr)
    centroid_hz = float(np.mean(centroid))

    t0 = i * SEGMENT_SECS
    t1 = t0 + SEGMENT_SECS
    print(
        f"  Seg {i+1:<4}  {t0}s – {t1}s      "
        f"{pitch_hz:>10.1f}   "
        f"{amplitude:>14.5f}   "
        f"{centroid_hz:>18.1f}"
    )

print()
print("Interpretation guide:")
print("  Pitch (Hz)          — higher = higher note; ~200 Hz = low vocal, ~500+ Hz = high")
print("  Amplitude (RMS)     — higher = louder")
print("  Spectral Centroid   — higher = brighter/breathier tone")
