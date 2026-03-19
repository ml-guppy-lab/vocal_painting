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

import os                        # used to build the artwork folder path
import queue                     # thread-safe queue to pass data between threads
import threading                 # to run audio capture in a separate background thread
import time                      # used for bloom burst timing
from typing import Optional, Tuple  # type hints only


import cv2                       # OpenCV: window management and all drawing ops
import numpy as np               # fast array math for the canvas and audio buffer
import sounddevice as sd         # opens the microphone input stream

from vocal import (
    SAMPLE_RATE,      # mic sample rate in Hz (e.g. 44100)
    FRAME_SAMPLES,    # how many samples per analysis frame
    CALLBACK_BLOCK,   # samples per sounddevice callback chunk
    extract_features, # function: raw audio frame → {pitch, amplitude, centroid}
)
from paint import (
    CANVAS_HEIGHT,    # canvas pixel height (800)
    CANVAS_WIDTH,     # canvas pixel width (1200)
    SMOOTH_WINDOW,    # number of frames to average for brush smoothing
    features_to_brush, # maps {pitch, amplitude, centroid} → {y, thickness, color}
    BrushSmoother,    # moving-average smoother to prevent jumpy brush movement
)

# OpenCV uses BGR (Blue, Green, Red) instead of the usual RGB.
# paint.py returns color names like "red" or "cyan", so we map them here.
_COLOR_BGR: dict[str, Tuple[int, int, int]] = {
    "indigo":  ( 75,   0, 130),
    "violet":  (238, 130, 238),
    "blue":    (255,   0,   0),  # pure blue in BGR = (255, 0, 0)
    "cyan":    (255, 255,   0),
    "green":   (  0, 200,   0),
    "yellow":  (  0, 255, 255),
    "orange":  (  0, 165, 255),
    "red":     (  0,   0, 255),  # pure red in BGR = (0, 0, 255)
}
_DEFAULT_BGR: Tuple[int, int, int] = (200, 200, 200)  # light grey fallback

def color_to_bgr(name: str) -> Tuple[int, int, int]:
    # Look up the name; return grey if it's somehow not in the table
    return _COLOR_BGR.get(name, _DEFAULT_BGR)


# ── Constants ─────────────────────────────────────────────────────────────────
ANGLE_SPEED         = 0.04   # how many radians the brush moves per audio frame (~1 full circle every 8 s)
BASE_RADIUS         = 150    # default distance (px) from canvas center when no voice input
RADIUS_RANGE        = 130    # max px the pitch can push the radius outward or inward from BASE_RADIUS
WOBBLE_AMP          = 50     # max px of petal-bump effect at loudest amplitude
PETAL_N             = 6      # how many bumps appear per full orbit (6 = 6 petals)
BLOOM_THICK_THRESH  = 25     # if brush thickness reaches this (out of 40), start counting loud time
BLOOM_HOLD_SECS     = 1.0    # how many seconds you must hold a loud note to trigger a bloom burst
BLOOM_DURATION      = 2.5    # how long (seconds) the bloom surge lasts after it fires
BLOOM_EXTRA         = 90     # peak extra radius (px) added during bloom, decays to 0 over BLOOM_DURATION
WINDOW_NAME    = "Vocal Painter  |  SPACE = clear  |  Q = quit"  # shown in the window title bar
BG_BGR         = (0, 0, 0)   # black background colour in BGR
ARTWORK_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artwork")  # saves next to this script


# ── Audio worker ──────────────────────────────────────────────────────────────

def _audio_worker(
    brush_q: "queue.Queue[dict]",   # output queue — sends brush dicts to the main thread
    stop_event: threading.Event,    # main thread sets this to tell us to shut down
    sr: int,                        # sample rate
    frame_samples: int,             # how many samples make one analysis frame
    smooth_window: int,             # moving-average window size for the smoother
) -> None:
    """Background thread: mic → features → brush dicts → queue."""
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()  # raw mic chunks land here from the callback
    smoother = BrushSmoother(window=smooth_window)       # smooths jumpy brush values
    buffer   = np.zeros(0, dtype=np.float32)             # accumulates mic samples until we have a full frame

    def _cb(indata, frames, time_info, status) -> None:
        # sounddevice calls this on every audio block (runs in a C thread, must be fast)
        if status:
            print(f"[audio] {status}", flush=True)  # print any underrun/overrun warnings
        audio_q.put(indata[:, 0].copy())  # grab channel 0 (mono) and queue it

    # open the microphone; the 'with' block keeps it open until we exit
    with sd.InputStream(
        samplerate=sr, channels=1, dtype="float32",
        blocksize=CALLBACK_BLOCK, callback=_cb,
    ):
        while not stop_event.is_set():  # keep running until main thread signals stop
            try:
                chunk = audio_q.get(timeout=0.3)  # wait up to 0.3 s for a new mic chunk
            except queue.Empty:
                continue  # nothing arrived yet, loop again

            # append the new chunk onto our rolling buffer
            buffer = np.concatenate([buffer, chunk])

            # process as many complete frames as the buffer holds
            while len(buffer) >= frame_samples:
                frame  = buffer[:frame_samples]   # take exactly one frame from the front
                buffer = buffer[frame_samples:]   # keep the remainder for next time
                features = extract_features(frame, sr)         # pitch, amplitude, centroid
                brush    = smoother.update(features_to_brush(features))  # map + smooth → brush dict
                brush_q.put(brush)  # send to main thread for drawing


# ── Canvas helpers ────────────────────────────────────────────────────────────

def _blank_canvas(h: int = CANVAS_HEIGHT, w: int = CANVAS_WIDTH) -> np.ndarray:
    """Create a fresh black painting canvas."""
    # np.zeros gives an all-black image; shape is (height, width, 3 color channels)
    return np.zeros((h, w, 3), dtype=np.uint8)


def _save_painting(painting: np.ndarray) -> None:
    """Save painting to ARTWORK_DIR with a sequential filename."""
    os.makedirs(ARTWORK_DIR, exist_ok=True)  # create artwork/ folder if it doesn't exist yet
    existing = [
        f for f in os.listdir(ARTWORK_DIR)   # look at every file in the folder
        if f.startswith("painting_") and f.endswith(".png")  # count only our own saved files
    ]
    next_num = len(existing) + 1  # next sequential number (1-based)
    filename = os.path.join(ARTWORK_DIR, f"painting_{next_num:03d}.png")  # e.g. painting_003.png
    cv2.imwrite(filename, painting)  # write the numpy array to disk as a PNG
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
    h, w = CANVAS_HEIGHT, CANVAS_WIDTH  # unpack canvas dimensions for convenience

    # ── painting layer ────────────────────────────────────────────────────────
    painting = _blank_canvas(h, w)  # start with a fresh all-black canvas

    # ── audio thread ──────────────────────────────────────────────────────────
    brush_q    : "queue.Queue[dict]" = queue.Queue()  # bridge between audio thread and draw loop
    stop_event = threading.Event()   # we'll set this flag when it's time to stop the audio thread
    audio_thread = threading.Thread(
        target=_audio_worker,        # function the thread will run
        args=(brush_q, stop_event, sr, frame_samples, smooth_window),
        daemon=True,                 # daemon=True means it dies automatically if the main program exits
    )
    audio_thread.start()  # kick off the background mic capture

    # create a resizable OpenCV window with the given title
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)  # set initial window size to match canvas

    # ── draw state ────────────────────────────────────────────────────────────
    cx, cy          = w // 2, h // 2   # orbit center = exact middle of canvas
    angle           = 0.0              # brush starts at angle 0 (rightmost point of orbit)
    amp_loud_since  = None             # records when the voice first got loud enough to arm bloom
    bloom_end_time  = 0.0              # wall-clock time when the current bloom burst expires
    prev_pt: Optional[Tuple[int, int]] = None  # last drawn point; None = pen is lifted

    # ── console ───────────────────────────────────────────────────────────────
    print(f"  Canvas   : {w} × {h} px")
    print("  SPACE = clear canvas  |  Q / ESC = quit")
    print("  Sing or hum — watch the painting grow!\n")

    try:
        while True:
            # ── keyboard ─────────────────────────────────────────────────────
            # waitKey(1) processes window events and returns the key pressed (or -1).
            # & 0xFF masks to the lowest 8 bits so it works correctly on all platforms.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC → quit
                break
            if key == ord(" "):  # SPACE → wipe the canvas and reset all state
                painting        = _blank_canvas(h, w)
                prev_pt         = None   # lift the pen so no line connects to old position
                angle           = 0.0    # restart orbit from angle 0
                amp_loud_since  = None   # cancel any in-progress bloom arming
                bloom_end_time  = 0.0    # cancel any active bloom burst
                print("  [SPACE] Canvas cleared.", flush=True)

            # ── check window still open ───────────────────────────────────────
            # getWindowProperty returns < 1 if the user closed the window with the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # ── drain brush queue ─────────────────────────────────────────────
            # get_nowait() grabs a brush dict without blocking.
            # If the audio thread hasn't produced one yet, we just redisplay and wait.
            try:
                brush = brush_q.get_nowait()
            except queue.Empty:
                cv2.imshow(WINDOW_NAME, painting)  # keep the window alive even during silence
                continue

            # ── radius = base + pitch + wobble + bloom ─────────────────────────
            now   = time.time()              # current wall-clock time (for bloom timing)
            bgr   = color_to_bgr(brush["color"])   # convert color name → BGR tuple
            thick = brush["thickness"]       # stroke width in pixels (driven by amplitude)

            # pitch_var: high pitch (small y) → positive → radius grows outward
            #            low pitch  (large y) → negative → radius shrinks inward
            pitch_var = int(np.interp(brush["y"], [0, h], [RADIUS_RANGE, -RADIUS_RANGE]))

            # wobble: a sine wave with PETAL_N cycles per orbit, scaled by how loud we are.
            # Louder voice = deeper bumps = more defined petal edges.
            amp_t  = np.clip(thick / 40.0, 0.0, 1.0)  # normalise thickness to 0..1
            wobble = int(amp_t * WOBBLE_AMP * np.sin(angle * PETAL_N))

            # bloom burst logic:
            # Step 1 — arm: start a timer the moment voice gets loud enough
            # Step 2 — fire: if we've been loud for BLOOM_HOLD_SECS, trigger the burst
            if thick >= BLOOM_THICK_THRESH:
                if amp_loud_since is None:
                    amp_loud_since = now  # start the loud timer
                elif (now - amp_loud_since >= BLOOM_HOLD_SECS
                        and now >= bloom_end_time):  # don't fire if a burst is already active
                    bloom_end_time = now + BLOOM_DURATION  # set burst expiry
                    amp_loud_since = None   # re-arm so next loud note can fire again
                    print("  [BLOOM] Burst triggered!", flush=True)
            else:
                amp_loud_since = None  # voice went quiet — reset the loud timer

            # bloom_boost decays linearly from BLOOM_EXTRA → 0 over BLOOM_DURATION seconds
            bloom_boost = 0
            if now < bloom_end_time:
                t           = (bloom_end_time - now) / BLOOM_DURATION  # goes from 1 → 0
                bloom_boost = int(BLOOM_EXTRA * t)

            # final radius: never go below 5 px so we always draw something
            radius = max(5, BASE_RADIUS + pitch_var + wobble + bloom_boost)

            # ── draw stroke ───────────────────────────────────────────────────
            # convert polar coordinates (radius, angle) to canvas pixel coordinates
            curr_pt = (
                cx + int(radius * np.cos(angle)),  # x = center_x + r * cos(θ)
                cy + int(radius * np.sin(angle)),  # y = center_y + r * sin(θ)
            )

            if prev_pt is not None:
                # draw a line from the last point to the current point (continuous stroke)
                cv2.line(painting, prev_pt, curr_pt, bgr, thick, cv2.LINE_AA)
            else:
                # pen was lifted (first frame, or after SPACE) — draw a single dot to start
                cv2.circle(painting, curr_pt, max(1, thick // 2), bgr, -1, cv2.LINE_AA)

            prev_pt = curr_pt    # remember current point for next frame's line
            angle  += ANGLE_SPEED  # advance the brush along the orbit

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
        # 1. Tell the audio thread to stop, then wait for it to finish cleanly
        stop_event.set()
        audio_thread.join(timeout=2.0)

        # 2. Save the final painting to the artwork/ folder
        _save_painting(painting)

        # 3. Close the OpenCV window.
        #    On macOS, destroyAllWindows alone is not enough — we must pump
        #    the event queue a few extra times with waitKey(1) to flush it.
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        print("\n  Painting finished.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_painter()

