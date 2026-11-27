"""Blink detection using the Eye Aspect Ratio (EAR).

EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|) over six eyelid landmarks
ordered [outer corner, upper1, upper2, inner corner, lower2, lower1]. An open
eye sits around 0.25-0.35; it drops towards 0.1 when closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .landmarks import LEFT_EYE_EAR, RIGHT_EYE_EAR


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR for a (6, 2) array of eyelid points."""
    if points.shape != (6, 2):
        raise ValueError("eye_aspect_ratio expects a (6, 2) array")
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def ear_from_landmarks(landmarks: np.ndarray, indices: Sequence[int] = RIGHT_EYE_EAR) -> float:
    return eye_aspect_ratio(landmarks[list(indices)])


def both_eyes_ear(landmarks: np.ndarray) -> tuple[float, float]:
    """(right_ear, left_ear) from a full (478, 2) landmark array."""
    return ear_from_landmarks(landmarks, RIGHT_EYE_EAR), ear_from_landmarks(landmarks, LEFT_EYE_EAR)


@dataclass
class BlinkEvent:
    count: int
    duration_frames: int
    is_blink: bool = True


class BlinkDetector:
    """Counts a blink when the averaged EAR dips below ``threshold`` for at
    least ``min_frames`` consecutive frames and then recovers.

    The eye is considered open again only once EAR rises above
    ``threshold + reopen_margin``, which stops a half-closed eye hovering at
    the threshold from registering a burst of blinks. Closures longer than
    ``max_frames`` are reported with ``is_blink=False`` (eyes held shut, or the
    face tilted so far down that the lids read as closed) and are not counted.

    ``update`` returns a ``BlinkEvent`` on the frame the eye reopens, else None.
    """

    def __init__(
        self,
        threshold: float = 0.21,
        min_frames: int = 2,
        max_frames: int = 15,
        reopen_margin: float = 0.03,
    ) -> None:
        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.reopen_margin = reopen_margin
        self.count = 0
        self.closed_frames = 0
        self.last_ear = 0.0

    @property
    def is_closed(self) -> bool:
        return self.closed_frames > 0

    def reset(self) -> None:
        self.closed_frames = 0

    def update(self, ear: float) -> BlinkEvent | None:
        self.last_ear = ear
        closing = ear < self.threshold
        staying_closed = self.closed_frames > 0 and ear < self.threshold + self.reopen_margin
        if closing or staying_closed:
            self.closed_frames += 1
            return None

        event = None
        if self.closed_frames >= self.min_frames:
            is_blink = self.closed_frames <= self.max_frames
            if is_blink:
                self.count += 1
            event = BlinkEvent(count=self.count, duration_frames=self.closed_frames, is_blink=is_blink)
        self.closed_frames = 0
        return event

    def update_from_landmarks(self, landmarks: np.ndarray) -> BlinkEvent | None:
        right, left = both_eyes_ear(landmarks)
        return self.update((right + left) / 2.0)
