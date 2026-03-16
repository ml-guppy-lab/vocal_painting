"""
paint.py — Phase 3: Mapping Voice Numbers → Brush Commands
-----------------------------------------------------------
Imports live feature extraction from vocal.py, maps each feature
to a paint property, applies a moving-average smoother, and prints
human-readable brush instructions in real time.

Mapping rules
─────────────
  pitch (Hz)              → Y position   high note  = top of canvas  (small Y)
  amplitude (RMS)         → thickness    loud       = fat brush
  spectral centroid (Hz)  → color hue    bright     = warm (red/orange)
                                         dull       = cool (blue/violet)

All logic lives in functions so this module can be imported by
a future drawing module without changes.
"""

import queue
import time
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from vocal import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    CALLBACK_BLOCK,
    extract_features,
)

# ── Canvas / brush constants ──────────────────────────────────────────────────
CANVAS_HEIGHT   = 800    # pixels (logical; no window opened yet)
CANVAS_WIDTH    = 1200   # pixels

PITCH_MIN       = 80     # Hz  — maps to bottom of canvas
PITCH_MAX       = 800    # Hz  — maps to top of canvas

THICKNESS_MIN   = 1      # px  — maps to silence
THICKNESS_MAX   = 40     # px  — maps to loudest expected signal
AMP_MIN         = 0.0
AMP_MAX         = 0.15   # RMS ceiling (clip louder signals to this)

CENTROID_MIN    = 500    # Hz  — cool/dull  → blue end
CENTROID_MAX    = 4000   # Hz  — bright     → red/orange end

SMOOTH_WINDOW   = 6      # frames in the moving-average window (~1 s)


# ── Color palette ─────────────────────────────────────────────────────────────
# Ordered from cool (low centroid) → warm (high centroid)
COLOR_STOPS = [
    (0.00, "indigo"),
    (0.15, "violet"),
    (0.30, "blue"),
    (0.50, "cyan"),
    (0.65, "green"),
    (0.78, "yellow"),
    (0.88, "orange"),
    (1.00, "red"),
]


# ── Mapping functions ─────────────────────────────────────────────────────────

def pitch_to_y(pitch_hz: float, canvas_height: int = CANVAS_HEIGHT) -> int:
    """
    Map pitch (Hz) → Y pixel on canvas.
    High note → small Y (top).  Low note → large Y (bottom).
    NaN (unvoiced) → centre of canvas.
    """
    if np.isnan(pitch_hz):
        return canvas_height // 2
    t = (pitch_hz - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)
    t = float(np.clip(t, 0.0, 1.0))
    return int((1.0 - t) * canvas_height)   # invert: high pitch = low Y


def amplitude_to_thickness(amplitude: float) -> int:
    """Map RMS amplitude → brush thickness in pixels."""
    t = (amplitude - AMP_MIN) / (AMP_MAX - AMP_MIN)
    t = float(np.clip(t, 0.0, 1.0))
    return max(1, int(THICKNESS_MIN + t * (THICKNESS_MAX - THICKNESS_MIN)))


def centroid_to_color(centroid_hz: float) -> str:
    """
    Map spectral centroid → color name using the COLOR_STOPS palette.
    Nearest stop by normalised distance.
    """
    t = (centroid_hz - CENTROID_MIN) / (CENTROID_MAX - CENTROID_MIN)
    t = float(np.clip(t, 0.0, 1.0))

    best_name = COLOR_STOPS[0][1]
    best_dist = abs(t - COLOR_STOPS[0][0])
    for stop_t, name in COLOR_STOPS[1:]:
        d = abs(t - stop_t)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def features_to_brush(features: dict) -> dict:
    """
    Convert a raw feature dict (from vocal.extract_features) into
    a brush-command dict ready for display or a drawing back-end.

    Returns
    -------
    {
        "y"        : int   — pixel row on canvas,
        "thickness": int   — brush width in pixels,
        "color"    : str   — color name,
        "pitch_hz" : float — raw pitch (for debug),
        "amp_rms"  : float — raw amplitude (for debug),
        "centroid" : float — raw centroid (for debug),
    }
    """
    return {
        "y":         pitch_to_y(features["pitch_hz"]),
        "thickness": amplitude_to_thickness(features["amplitude_rms"]),
        "color":     centroid_to_color(features["spectral_centroid_hz"]),
        "pitch_hz":  features["pitch_hz"],
        "amp_rms":   features["amplitude_rms"],
        "centroid":  features["spectral_centroid_hz"],
    }


# ── Smoother ──────────────────────────────────────────────────────────────────

class BrushSmoother:
    """
    Keeps a rolling window of recent brush commands and returns
    a moving-average blend so the brush doesn't jump erratically.
    """

    def __init__(self, window: int = SMOOTH_WINDOW) -> None:
        self._y         = deque(maxlen=window)
        self._thickness = deque(maxlen=window)

    def update(self, brush: dict) -> dict:
        """Push a new brush command and return the smoothed version."""
        self._y.append(brush["y"])
        self._thickness.append(brush["thickness"])

        smoothed = brush.copy()
        smoothed["y"]         = int(np.mean(self._y))
        smoothed["thickness"] = max(1, int(np.mean(self._thickness)))
        return smoothed


# ── Display ───────────────────────────────────────────────────────────────────

def print_brush(brush: dict) -> None:
    """Print a single brush-command line to the console."""
    pitch_str = f"{brush['pitch_hz']:6.1f} Hz" if not np.isnan(brush["pitch_hz"]) else " (silent)"
    print(
        f"  Brush at Y={brush['y']:>4}  |  "
        f"thickness={brush['thickness']:>2}  |  "
        f"color={brush['color']:<8}  "
        f"  [pitch={pitch_str}, amp={brush['amp_rms']:.4f}, centroid={brush['centroid']:.0f} Hz]",
        flush=True,
    )


# ── Live painting loop ────────────────────────────────────────────────────────

def live_paint_loop(
    duration: Optional[float] = None,
    sr: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    smooth_window: int = SMOOTH_WINDOW,
) -> None:
    """
    Stream mic → extract features → map to brush → smooth → print.

    Parameters
    ----------
    duration      : seconds to run (None = until Ctrl+C)
    sr            : sample rate in Hz
    frame_samples : audio samples per analysis frame
    smooth_window : number of frames in the moving average
    """
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    smoother = BrushSmoother(window=smooth_window)

    def _callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[stream] {status}", flush=True)
        audio_q.put(indata[:, 0].copy())

    print("=" * 72)
    print("  Vocal Painter — Phase 3: Mapping Voice → Brush Commands")
    print("=" * 72)
    print(f"  Canvas : {CANVAS_WIDTH} × {CANVAS_HEIGHT} px  |  Smooth window : {smooth_window} frames")
    print("  Mapping: pitch→Y  |  amplitude→thickness  |  centroid→color")
    if duration:
        print(f"  Running for {duration} s — press Ctrl+C to stop early.")
    else:
        print("  Sing or hum — press Ctrl+C to stop.")
    print()
    print(f"  Try: loud vs soft → watch 'thickness' change")
    print(f"       high vs low  → watch 'Y' jump")
    print()

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

                    features = extract_features(frame, sr)
                    raw_brush = features_to_brush(features)
                    smooth_brush = smoother.update(raw_brush)
                    print_brush(smooth_brush)

        except KeyboardInterrupt:
            print("\n  Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    live_paint_loop()
