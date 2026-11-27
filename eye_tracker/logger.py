"""Movement log: prints to the terminal on change, optionally mirrors to CSV."""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class MovementLogger:
    """Emits one line per *event* rather than per frame.

    Events: gaze direction change, blink, face lost / face found, mouse click.
    """

    def __init__(self, csv_path: Optional[str] = None, stream: IO[str] = sys.stdout, quiet: bool = False) -> None:
        self._stream = stream
        self._quiet = quiet
        self._last_direction: Optional[str] = None
        self._face_visible: Optional[bool] = None
        self._csv_file: Optional[IO[str]] = None
        self._csv: Optional[csv.writer] = None
        self._t0 = time.monotonic()
        if csv_path:
            path = Path(csv_path)
            new_file = not path.exists() or path.stat().st_size == 0
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(path, "a", newline="")
            self._csv = csv.writer(self._csv_file)
            if new_file:
                self._csv.writerow(["wall_time", "elapsed_s", "event", "direction", "h", "v", "ear", "blink_count"])

    def _emit(self, text: str, event: str, direction: str = "", h=None, v=None, ear=None, blinks=None) -> None:
        if not self._quiet:
            self._stream.write(f"{_now()}  {text}\n")
            self._stream.flush()
        if self._csv is not None:
            fmt = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
            self._csv.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                f"{time.monotonic() - self._t0:.3f}",
                event, direction, fmt(h), fmt(v), fmt(ear), "" if blinks is None else blinks,
            ])
            self._csv_file.flush()

    def gaze(self, direction: str, h: float, v: float, ear: Optional[float] = None, blinks: Optional[int] = None) -> bool:
        """Log if the direction changed. Returns True when a line was written."""
        if direction == self._last_direction:
            return False
        self._last_direction = direction
        self._emit(f"gaze={direction:<7} h={h:.2f} v={v:+.2f}", "gaze", direction, h, v, ear, blinks)
        return True

    def blink(self, count: int, duration_frames: int, ear: Optional[float] = None) -> None:
        self._emit(f"BLINK #{count} ({duration_frames} frames)", "blink", self._last_direction or "", ear=ear, blinks=count)

    def eyes_closed(self, duration_frames: int, ear: Optional[float] = None, blinks: Optional[int] = None) -> None:
        self._emit(f"eyes closed for {duration_frames} frames", "eyes_closed", self._last_direction or "", ear=ear, blinks=blinks)

    def click(self, button: str, x: int, y: int, blinks: Optional[int] = None) -> None:
        self._emit(f"CLICK {button} at ({x}, {y})", f"click_{button}", self._last_direction or "", blinks=blinks)

    def face(self, visible: bool) -> None:
        if visible == self._face_visible:
            return
        self._face_visible = visible
        if visible:
            self._emit("face detected", "face_found")
        else:
            self._last_direction = None
            self._emit("face lost", "face_lost")

    def info(self, text: str) -> None:
        if not self._quiet:
            self._stream.write(f"{_now()}  {text}\n")
            self._stream.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv = None
