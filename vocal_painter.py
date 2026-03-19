"""
vocal_painter.py — OpenCV Canvas
---------------------------------
Paints in real time from microphone input on a black OpenCV canvas.

Architecture
────────────
  audio thread  →  brush_q  →  main thread (OpenCV draw loop)

Drawing model
─────────────
  • Painting layer  : numpy uint8 black canvas; strokes accumulate.
  • Radius = base + pitch_var + amplitude_wobble + bloom_boost.
  • Wobble: sin(angle × PETAL_N) scaled by amplitude → petal-edge bumps.
  • Bloom burst: hold a loud note for 1 s → radius surges for 2.5 s.
  • Pen size follows amplitude  (loud = thick).
  • Pen color follows spectral centroid  (bright = warm, dull = cool).

Controls
────────
  SPACE  — clear the canvas
  Q / ESC / close button — quit
"""

import os
import queue
import threading
import time
from typing import Optional, Tuple


import cv2
import numpy as np
import sounddevice as sd

from vocal import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    CALLBACK_BLOCK,
    extract_features,
)
from paint import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    SMOOTH_WINDOW,
    features_to_brush,
    BrushSmoother,
)

# ── Color mapping: paint.py name → OpenCV BGR ────────────────────────────────
_COLOR_BGR: dict[str, Tuple[int, int, int]] = {
    "indigo":  ( 75,   0, 130),
    "violet":  (238, 130, 238),
    "blue":    (255,   0,   0),
    "cyan":    (255, 255,   0),
    "green":   (  0, 200,   0),
    "yellow":  (  0, 255, 255),
    "orange":  (  0, 165, 255),
    "red":     (  0,   0, 255),
}
_DEFAULT_BGR: Tuple[int, int, int] = (200, 200, 200)

def color_to_bgr(name: str) -> Tuple[int, int, int]:
    return _COLOR_BGR.get(name, _DEFAULT_BGR)


# ── Constants ─────────────────────────────────────────────────────────────────
ANGLE_SPEED         = 0.04   # radians advanced per brush frame (~1 full orbit / 8 s)
BASE_RADIUS         = 150    # px — resting orbit radius
RADIUS_RANGE        = 130    # px — max bloom out / shrink in from base (pitch)
WOBBLE_AMP          = 50     # px — max petal-edge wobble at full amplitude
PETAL_N             = 6      # sine frequency multiplier — N bumps per orbit
BLOOM_THICK_THRESH  = 25     # brush thickness (out of 40) that arms the bloom
BLOOM_HOLD_SECS     = 1.0    # seconds of sustained loud voice to fire a bloom burst
BLOOM_DURATION      = 2.5    # seconds the bloom surge lasts
BLOOM_EXTRA         = 90     # px added at peak bloom, decays linearly to 0
WINDOW_NAME    = "Vocal Painter  |  SPACE = clear  |  Q = quit"
BG_BGR         = (0, 0, 0)      # pure black background fill
ARTWORK_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artwork")


# ── Audio worker ──────────────────────────────────────────────────────────────

def _audio_worker(
    brush_q: "queue.Queue[dict]",
    stop_event: threading.Event,
    sr: int,
    frame_samples: int,
    smooth_window: int,
) -> None:
    """Background thread: mic → features → brush dicts → queue."""
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    smoother = BrushSmoother(window=smooth_window)
    buffer   = np.zeros(0, dtype=np.float32)

    def _cb(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sr, channels=1, dtype="float32",
        blocksize=CALLBACK_BLOCK, callback=_cb,
    ):
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.3)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, chunk])
            while len(buffer) >= frame_samples:
                frame  = buffer[:frame_samples]
                buffer = buffer[frame_samples:]
                features = extract_features(frame, sr)
                brush    = smoother.update(features_to_brush(features))
                brush_q.put(brush)


# ── Canvas helpers ────────────────────────────────────────────────────────────

def _blank_canvas(h: int = CANVAS_HEIGHT, w: int = CANVAS_WIDTH) -> np.ndarray:
    """Create a fresh black painting canvas."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _save_painting(painting: np.ndarray) -> None:
    """Save painting to ARTWORK_DIR with a sequential filename."""
    os.makedirs(ARTWORK_DIR, exist_ok=True)
    existing = [
        f for f in os.listdir(ARTWORK_DIR)
        if f.startswith("painting_") and f.endswith(".png")
    ]
    next_num = len(existing) + 1
    filename = os.path.join(ARTWORK_DIR, f"painting_{next_num:03d}.png")
    cv2.imwrite(filename, painting)
    print(f"  Painting saved → {filename}")


# ── Main painter ──────────────────────────────────────────────────────────────

def run_painter(
    sr: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    smooth_window: int = SMOOTH_WINDOW,
) -> None:
    """
    Open the OpenCV window and paint in real time from mic input.

    Parameters
    ----------
    sr            : audio sample rate in Hz
    frame_samples : samples per analysis frame
    smooth_window : moving-average window for brush smoothing
    """
    h, w = CANVAS_HEIGHT, CANVAS_WIDTH

    # ── painting layer ────────────────────────────────────────────────────────
    painting = _blank_canvas(h, w)

    # ── audio thread ──────────────────────────────────────────────────────────
    brush_q    : "queue.Queue[dict]" = queue.Queue()
    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=_audio_worker,
        args=(brush_q, stop_event, sr, frame_samples, smooth_window),
        daemon=True,
    )
    audio_thread.start()

    # ── OpenCV window ─────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    # ── draw state ────────────────────────────────────────────────────────────
    cx, cy          = w // 2, h // 2   # orbit center
    angle           = 0.0              # current angle in radians
    amp_loud_since  = None             # timestamp when loud threshold was crossed
    bloom_end_time  = 0.0              # timestamp when bloom burst expires
    prev_pt: Optional[Tuple[int, int]] = None

    # ── console ───────────────────────────────────────────────────────────────
    print(f"  Canvas   : {w} × {h} px")
    print("  SPACE = clear canvas  |  Q / ESC = quit")
    print("  Sing or hum — watch the painting grow!\n")

    try:
        while True:
            # ── keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            if key == ord(" "):
                painting        = _blank_canvas(h, w)
                prev_pt         = None
                angle           = 0.0
                amp_loud_since  = None
                bloom_end_time  = 0.0
                print("  [SPACE] Canvas cleared.", flush=True)

            # ── check window still open ───────────────────────────────────────
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # ── drain brush queue ─────────────────────────────────────────────
            try:
                brush = brush_q.get_nowait()
            except queue.Empty:
                cv2.imshow(WINDOW_NAME, painting)
                continue

            # ── radius = base + pitch + wobble + bloom ─────────────────────────
            now   = time.time()
            bgr   = color_to_bgr(brush["color"])
            thick = brush["thickness"]

            pitch_var = int(np.interp(brush["y"], [0, h], [RADIUS_RANGE, -RADIUS_RANGE]))

            # petal wobble: N bumps per orbit, depth scaled by amplitude
            amp_t  = np.clip(thick / 40.0, 0.0, 1.0)
            wobble = int(amp_t * WOBBLE_AMP * np.sin(angle * PETAL_N))

            # bloom burst: arm when loud, fire after BLOOM_HOLD_SECS
            if thick >= BLOOM_THICK_THRESH:
                if amp_loud_since is None:
                    amp_loud_since = now
                elif (now - amp_loud_since >= BLOOM_HOLD_SECS
                        and now >= bloom_end_time):
                    bloom_end_time = now + BLOOM_DURATION
                    amp_loud_since = None   # re-arm for next burst
                    print("  [BLOOM] Burst triggered!", flush=True)
            else:
                amp_loud_since = None

            bloom_boost = 0
            if now < bloom_end_time:
                t           = (bloom_end_time - now) / BLOOM_DURATION  # 1 → 0
                bloom_boost = int(BLOOM_EXTRA * t)

            radius = max(5, BASE_RADIUS + pitch_var + wobble + bloom_boost)

            # ── draw stroke ───────────────────────────────────────────────────
            curr_pt = (
                cx + int(radius * np.cos(angle)),
                cy + int(radius * np.sin(angle)),
            )

            if prev_pt is not None:
                cv2.line(painting, prev_pt, curr_pt, bgr, thick, cv2.LINE_AA)
            else:
                cv2.circle(painting, curr_pt, max(1, thick // 2), bgr, -1, cv2.LINE_AA)

            prev_pt = curr_pt
            angle  += ANGLE_SPEED

            # ── display ───────────────────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, painting)

            print(
                f"  r={radius:>4}  thick={thick:>2}  "
                f"color={brush['color']:<8}  angle={angle:>6.2f}"
                + ("  🌸 BLOOM" if now < bloom_end_time else ""),
                flush=True,
            )

    except KeyboardInterrupt:
        pass

    finally:
        # 1. Signal audio thread to stop
        stop_event.set()
        audio_thread.join(timeout=2.0)

        # 2. Save the painting
        _save_painting(painting)

        # 3. Destroy window + flush macOS event queue (required on macOS —
        #    destroyAllWindows alone doesn't process the close event)
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        print("\n  Painting finished.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_painter()

