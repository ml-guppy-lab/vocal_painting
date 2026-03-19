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
  • X advances left→right at a fixed per-frame speed then wraps.
  • Y driven by pitch  (high = top, low = bottom).
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
DRAW_DURATION  = 30      # seconds; 0 = run until window is closed / Q pressed
X_SPEED        = 8       # pixels advanced along X per brush frame
X_MARGIN       = 20      # pixels kept clear at each horizontal edge
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
    duration: float = DRAW_DURATION,
    sr: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    smooth_window: int = SMOOTH_WINDOW,
) -> None:
    """
    Open the OpenCV window and paint in real time from mic input.

    Parameters
    ----------
    duration      : seconds to paint (0 = until Q / window closed)
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
    x_left       = X_MARGIN
    x_right      = w - X_MARGIN
    current_x    = x_left
    prev_pt: Optional[Tuple[int, int]] = None

    # ── console ───────────────────────────────────────────────────────────────
    dur_str = f"{duration} s" if duration > 0 else "until Q / window close"
    print("=" * 64)
    print("  Vocal Painter — Phase 5: OpenCV Canvas")
    print("=" * 64)
    print(f"  Duration : {dur_str}")
    print(f"  Canvas   : {w} × {h} px")
    print("  SPACE = clear canvas  |  Q / ESC = quit")
    print("  Sing or hum — watch the painting grow!\n")

    start_time = time.time()

    try:
        while True:
            # ── quit conditions ───────────────────────────────────────────────
            if duration > 0 and (time.time() - start_time) >= duration:
                break

            # ── keyboard ─────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            if key == ord(" "):
                painting = _blank_canvas(h, w)
                prev_pt  = None
                current_x = x_left
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

            # ── wrap X ────────────────────────────────────────────────────────
            if current_x >= x_right:
                current_x = x_left
                prev_pt   = None   # lift pen at wrap

            # ── draw stroke ───────────────────────────────────────────────────
            curr_pt = (current_x, brush["y"])
            bgr     = color_to_bgr(brush["color"])
            thick   = brush["thickness"]

            if prev_pt is not None:
                cv2.line(painting, prev_pt, curr_pt, bgr, thick, cv2.LINE_AA)
            else:
                cv2.circle(painting, curr_pt, max(1, thick // 2), bgr, -1, cv2.LINE_AA)

            prev_pt    = curr_pt
            current_x += X_SPEED

            # ── display ───────────────────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, painting)

            print(
                f"  y={brush['y']:>4}  thick={thick:>2}  "
                f"color={brush['color']:<8}  x={current_x:>5}",
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

