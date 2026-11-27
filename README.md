# Webcam Eye Tracker

A Python desktop app that watches you through the webcam and tracks your eye
movement in real time. It runs entirely on your machine with OpenCV and
MediaPipe's Face Landmarker (478 landmarks, including the iris); no server,
no calibration.

In the window you get:

- Eye contours and iris centres drawn on both eyes
- A gaze direction label (`CENTER`, `LEFT`, `RIGHT`, `UP`, `DOWN`) and a small
  3x3 "radar" showing where the eyes are pointing
- Blink detection with a running blink counter
- FPS, and a clear "No face detected" banner when you leave the frame

In the terminal you get a movement log that only prints when something
changes, so it stays readable:

```
15:41:02.118  gaze=LEFT    h=0.37 v=+0.01
15:41:02.884  gaze=CENTER  h=0.49 v=+0.00
15:41:03.410  BLINK #3 (4 frames)
15:41:04.021  gaze=UP      h=0.50 v=-0.08
```

Directions are from *your* point of view: `LEFT` means you looked to your own
left.

## Requirements

- Python 3.10 - 3.12
- A webcam
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `pip`

## Install

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e .
```

On a minimal Linux install MediaPipe also needs the system GL libraries:
`sudo apt install libegl1 libgl1 libglib2.0-0`. macOS and Windows need nothing extra.

The first run downloads the Face Landmarker model (~3.7 MB) into `models/`.
If your machine has no internet access, download
[`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
yourself and pass `--model path/to/face_landmarker.task`.

## Run

```bash
uv run python -m eye_tracker
# or, after `uv sync`:
uv run eye-tracker
```

### macOS notes

- The first run triggers macOS's camera permission prompt for whichever app
  launched the terminal (Terminal, iTerm, Cursor, VS Code...). Click Allow. If
  you dismissed it, enable it under System Settings > Privacy & Security >
  Camera, then start the app again.
- If you get a black image or the wrong camera, your iPhone may be registered
  as a Continuity Camera at index 0. Try `--camera 1`.
- Apple Silicon is supported natively. The locked OpenCV build requires macOS 15
  or newer; on an older macOS run `uv lock --upgrade-package opencv-python
  --upgrade-package opencv-contrib-python` after adding an upper bound
  `opencv-python<5` to `pyproject.toml`. Intel Macs need MediaPipe 0.10.x
  (`uv add "mediapipe<1"`), which exposes the same `tasks` API.

Keys while the window is focused:

| Key | Action                         |
| --- | ------------------------------ |
| `q` / `Esc` | Quit                   |
| `f` | Toggle mirror (selfie) view    |
| `l` | Toggle landmark overlay        |
| `r` | Reset blink counter and smoothing |

Useful options:

```
--camera N          webcam index (default 0)
--video PATH        run on a video file instead of the webcam (played at its own frame rate)
--fast              with --video: do not pace playback, process as fast as possible
--width/--height    request a capture resolution
--no-flip           do not mirror the preview
--log-csv FILE      also append every event to a CSV file
--no-window         headless mode (log only), e.g. on a server or in CI
--save-frame FILE   write the last rendered frame to an image (handy with --no-window)
--max-frames N      stop after N frames
--model PATH        use a specific face_landmarker.task
--h-threshold X     horizontal sensitivity (default 0.08, smaller = more sensitive)
--up-threshold X    / --down-threshold X   vertical sensitivity (defaults 0.05 / 0.045)
--v-center X        vertical offset considered "centre" (default -0.08, see below)
--ear-threshold X   blink threshold on the Eye Aspect Ratio (default 0.21)
--smoothing X       EMA factor for gaze ratios, 0-1 (default 0.35, lower = smoother)
```

## How direction is decided

For each eye the iris centre (landmark 468 / 473) is projected onto the line
between the eye corners:

- `h` is the position along that line, 0 at the image-left corner and 1 at the
  image-right corner. Centre is 0.5.
- `v` is the offset perpendicular to that line, divided by the eye width.
  Negative is up, positive is down. With a neutral gaze the iris centre sits a
  little *above* the corner line, so "centre" defaults to `v = -0.08`. If the
  label leans `UP` or `DOWN` while you look straight at the camera, read the
  `v=` value on the HUD and pass it as `--v-center`.

Both eyes are averaged, exponentially smoothed, and compared against the
thresholds above. The larger excursion (relative to its threshold) wins, and a
small hysteresis band keeps the label from flickering at the boundary. While a
blink is detected the gaze label is held rather than recomputed.

Because everything is normalised by eye width, it works at different distances
from the camera. It does assume your head is roughly facing the camera; large
head turns read as gaze shifts, which is expected without calibration.

## Tests

```bash
uv run pytest
```

The tests exercise the gaze and blink maths on synthetic landmarks and need no
camera. To check the full pipeline without a webcam, run it on any video that
contains a face:

```bash
uv run python -m eye_tracker --video sample.mp4 --no-window --save-frame out.png
```

## Project layout

```
eye_tracker/
  __main__.py   CLI entry point and main loop
  camera.py     webcam / video file frame source
  landmarks.py  MediaPipe Face Landmarker wrapper + model download
  gaze.py       iris ratios -> direction (pure functions + smoother)
  blink.py      Eye Aspect Ratio -> blink events
  overlay.py    all OpenCV drawing
  logger.py     terminal and CSV movement log
tests/
  test_gaze.py
  test_blink.py
```

## Privacy

Frames never leave your machine. Nothing is recorded unless you pass
`--log-csv` or `--save-frame`.
