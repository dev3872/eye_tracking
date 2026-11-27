"""Frame sources: webcam or video file, with graceful failure."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the webcam or video file cannot be opened or read."""


@dataclass
class Frame:
    image: np.ndarray
    timestamp_ms: int
    index: int


class FrameSource:
    """Yields frames from a webcam index or a video file path.

    For a webcam, timestamps come from a monotonic clock. For a video file,
    timestamps are derived from the file's FPS so playback speed does not
    matter to the landmarker's VIDEO mode (which needs increasing timestamps).
    """

    def __init__(
        self,
        camera: int = 0,
        video: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        open_retries: int = 3,
    ) -> None:
        self.is_file = video is not None
        self._source = video if video is not None else camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = width
        self._height = height
        self._open_retries = open_retries
        self._start_monotonic = 0.0
        self._fps = 30.0
        self._last_ts = -1

    def open(self) -> "FrameSource":
        last_err = None
        for attempt in range(1, self._open_retries + 1):
            cap = cv2.VideoCapture(self._source)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
            last_err = f"attempt {attempt}/{self._open_retries} failed"
            if self.is_file:
                break
            time.sleep(0.5)
        if self._cap is None:
            if self.is_file:
                raise CameraError(f"Could not open video file: {self._source}")
            raise CameraError(
                f"Could not open webcam #{self._source} ({last_err}). "
                "Check that a camera is connected, not in use by another app, "
                "and that this program has camera permission."
            )

        if not self.is_file:
            if self._width:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self._fps = float(fps)
        self._start_monotonic = time.monotonic()
        return self

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise CameraError("FrameSource.open() must be called before frames()")
        index = 0
        consecutive_failures = 0
        while True:
            ok, image = self._cap.read()
            if not ok or image is None:
                if self.is_file:
                    return
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    raise CameraError("Webcam stopped delivering frames.")
                time.sleep(0.02)
                continue
            consecutive_failures = 0

            if self.is_file:
                ts = int(index * 1000.0 / self._fps)
            else:
                ts = int((time.monotonic() - self._start_monotonic) * 1000.0)
            # The landmarker requires strictly increasing timestamps.
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts

            yield Frame(image=image, timestamp_ms=ts, index=index)
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
