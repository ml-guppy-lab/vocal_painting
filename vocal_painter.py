"""
vocal_painter.py — Phase 4: Turtle Drawing
-------------------------------------------
Combines live vocal feature extraction (vocal.py) and brush mapping
(paint.py) to drive a turtle canvas in real time while you sing.

Architecture
────────────
  audio thread  →  queue  →  main thread (turtle)
  sounddevice callback fills a queue with smoothed brush dicts;
  the main loop drains it and moves the turtle without any blocking.

Drawing model
─────────────
  • X advances from left to right at a fixed per-frame speed.
  • Y is driven by pitch  (high note → top, low note → bottom).
  • Pen size follows amplitude  (loud → thick).
  • Pen color follows spectral centroid  (bright → warm, dull → cool).
  • When X reaches the right edge it wraps back to the left (new line).
"""

import queue
import threading
import time
import turtle
from typing import Optional

import numpy as np
import sounddevice as sd

from vocal import (
    SAMPLE_RATE,
    FRAME_SAMPLES,
    CALLBACK_BLOCK,
    extract_features,
)
from paint import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    SMOOTH_WINDOW,
    features_to_brush,
    BrushSmoother,
)

# paint_y (0 = top, CANVAS_HEIGHT = bottom) → turtle y (0 = centre, + = up)
def _to_turtle_y(paint_y: int, canvas_height: int = CANVAS_HEIGHT) -> float:
    return (canvas_height / 2.0) - paint_y


# ── Drawing constants ─────────────────────────────────────────────────────────
DRAW_DURATION  = 20      # seconds; 0 = run until window is closed
X_SPEED        = 10      # turtle pixels advanced per brush frame (~185 ms)
X_MARGIN       = 30      # pixels kept clear at each horizontal edge
BG_COLOR       = "black"


# ── Audio worker (background thread) ─────────────────────────────────────────

def _audio_worker(
    brush_q: "queue.Queue[dict]",
    stop_event: threading.Event,
    sr: int,
    frame_samples: int,
    smooth_window: int,
) -> None:
    """
    Capture mic → extract features → map to brush → push to queue.
    Runs entirely on a daemon thread so it never blocks the turtle loop.
    """
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    smoother = BrushSmoother(window=smooth_window)
    buffer   = np.zeros(0, dtype=np.float32)

    def _callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=CALLBACK_BLOCK,
        callback=_callback,
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


# ── Turtle helpers ────────────────────────────────────────────────────────────

def _setup_screen() -> turtle.Screen:
    screen = turtle.Screen()
    screen.setup(width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    screen.bgcolor(BG_COLOR)
    screen.title("Vocal Painter  —  sing to draw!")
    screen.tracer(0)   # manual refresh for smooth drawing
    return screen


def _make_pen() -> turtle.Turtle:
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.speed(0)
    pen.pencolor("white")
    pen.pensize(2)
    return pen


def _reset_pen_to_left(pen: turtle.Turtle) -> float:
    """Lift pen, jump to left edge at Y=0, put pen down. Returns new X."""
    x_start = -(CANVAS_WIDTH / 2) + X_MARGIN
    pen.penup()
    pen.goto(x_start, 0)
    pen.pendown()
    return x_start


# ── Main painter ──────────────────────────────────────────────────────────────

def run_painter(
    duration: float = DRAW_DURATION,
    sr: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    smooth_window: int = SMOOTH_WINDOW,
) -> None:
    """
    Open the turtle canvas and paint in real time from mic input.

    Parameters
    ----------
    duration      : seconds to paint (0 = until window is closed / Ctrl+C)
    sr            : audio sample rate in Hz
    frame_samples : samples per analysis frame
    smooth_window : moving-average window for brush smoothing
    """
    # ── turtle setup ──────────────────────────────────────────────────────────
    screen = _setup_screen()
    pen    = _make_pen()

    x_right_limit = (CANVAS_WIDTH / 2) - X_MARGIN
    current_x     = _reset_pen_to_left(pen)

    # ── start audio thread ────────────────────────────────────────────────────
    brush_q    : "queue.Queue[dict]" = queue.Queue()
    stop_event = threading.Event()

    audio_thread = threading.Thread(
        target=_audio_worker,
        args=(brush_q, stop_event, sr, frame_samples, smooth_window),
        daemon=True,
    )
    audio_thread.start()

    # ── console header ────────────────────────────────────────────────────────
    dur_str = f"{duration} s" if duration > 0 else "until closed"
    print("=" * 60)
    print("  Vocal Painter — Phase 4: Turtle Drawing")
    print("=" * 60)
    print(f"  Duration : {dur_str}")
    print(f"  Canvas   : {CANVAS_WIDTH} × {CANVAS_HEIGHT} px  |  BG: {BG_COLOR}")
    print("  Hum or sing — watch the line respond!")
    print("  Close the window or Ctrl+C to stop early.\n")

    start_time = time.time()

    try:
        while True:
            # ── time limit check ──────────────────────────────────────────────
            if duration > 0 and (time.time() - start_time) >= duration:
                break

            # ── drain the brush queue (non-blocking) ──────────────────────────
            try:
                brush = brush_q.get_nowait()
            except queue.Empty:
                screen.update()
                time.sleep(0.01)
                continue

            # ── wrap X when we hit the right edge ─────────────────────────────
            if current_x >= x_right_limit:
                current_x = _reset_pen_to_left(pen)

            # ── move turtle ───────────────────────────────────────────────────
            turtle_y = _to_turtle_y(brush["y"])
            pen.pensize(brush["thickness"])
            pen.pencolor(brush["color"])
            pen.goto(current_x, turtle_y)

            current_x += X_SPEED
            screen.update()

            print(
                f"  Brush at Y={brush['y']:>4}  "
                f"thickness={brush['thickness']:>2}  "
                f"color={brush['color']:<8}  "
                f"x_canvas={int(current_x):>5}",
                flush=True,
            )

    except (KeyboardInterrupt, turtle.Terminator):
        pass

    finally:
        stop_event.set()
        print("\n  Painting finished.  Close the turtle window to exit.")
        try:
            turtle.done()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_painter()
