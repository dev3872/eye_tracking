"""Gaze direction from iris position, as pure functions plus a small smoother.

Conventions
-----------
* ``h`` is the horizontal iris position inside the eye in *image* coordinates:
  0.0 = iris touching the image-left corner, 1.0 = image-right corner, 0.5 = centred.
* ``v`` is the vertical iris offset from the line joining the eye corners,
  normalised by eye width. Negative = iris above the corner line (looking up),
  positive = below (looking down). Normalising by width rather than eyelid
  opening keeps the value stable while the lids move.
* Directions are reported from the *user's* point of view ("LEFT" = the user
  looked to their own left). ``mirrored`` tells the classifier whether the
  landmarks came from a horizontally flipped (selfie-view) frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS_CENTER,
)

CENTER = "CENTER"
LEFT = "LEFT"
RIGHT = "RIGHT"
UP = "UP"
DOWN = "DOWN"
DIRECTIONS = (CENTER, LEFT, RIGHT, UP, DOWN)


@dataclass(frozen=True)
class GazeThresholds:
    """How far the iris must travel before a direction is reported.

    ``horizontal`` is a deviation from h = 0.5; ``up`` / ``down`` are limits on
    ``v``. ``hysteresis`` is the fraction of a threshold the value must fall
    back inside before returning to CENTER, which stops flicker at the edge.
    """

    horizontal: float = 0.08
    up: float = 0.05
    down: float = 0.045
    # With a neutral gaze the iris centre sits slightly *above* the line
    # joining the eye corners (about 8% of the eye width), so "centre" is not
    # v = 0. Measured on real footage; override with --v-center if needed.
    vertical_center: float = -0.08
    hysteresis: float = 0.6


@dataclass(frozen=True)
class GazeMeasure:
    h: float
    v: float
    eye_width: float


def eye_measure(
    landmarks: np.ndarray,
    corners: Sequence[int],
    iris_center: int,
) -> Optional[GazeMeasure]:
    """Compute (h, v) for one eye. Returns None if the eye is degenerate."""
    a = landmarks[corners[0]]
    b = landmarks[corners[1]]
    iris = landmarks[iris_center]

    left_corner, right_corner = (a, b) if a[0] <= b[0] else (b, a)
    width = float(np.hypot(*(right_corner - left_corner)))
    if width < 1e-6:
        return None

    # Project the iris onto the corner-to-corner axis so a tilted head does not
    # read as a horizontal gaze shift.
    axis = (right_corner - left_corner) / width
    rel = iris - left_corner
    along = float(np.dot(rel, axis))
    # For an axis pointing image-right this normal is (0, 1): image-down, so
    # positive `across` means the iris sits below the corner line.
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    across = float(np.dot(rel, normal))

    h = along / width
    v = across / width
    # When the head turns far enough that an eye is nearly edge-on, the iris
    # landmarks collapse outside the corners. Treat that eye as unusable.
    if not -0.2 <= h <= 1.2:
        return None
    return GazeMeasure(h=h, v=v, eye_width=width)


def measure_gaze(landmarks: np.ndarray) -> Optional[GazeMeasure]:
    """Average the per-eye measures. Returns None if neither eye is usable."""
    right = eye_measure(landmarks, RIGHT_EYE_CORNERS, RIGHT_IRIS_CENTER)
    left = eye_measure(landmarks, LEFT_EYE_CORNERS, LEFT_IRIS_CENTER)
    usable = [m for m in (right, left) if m is not None]
    if not usable:
        return None
    h = sum(m.h for m in usable) / len(usable)
    v = sum(m.v for m in usable) / len(usable)
    w = sum(m.eye_width for m in usable) / len(usable)
    return GazeMeasure(h=h, v=v, eye_width=w)


def classify(
    h: float,
    v: float,
    thresholds: GazeThresholds = GazeThresholds(),
    mirrored: bool = True,
    current: str = CENTER,
) -> str:
    """Map (h, v) to a direction, with hysteresis relative to ``current``.

    Horizontal and vertical excursions are compared as fractions of their own
    threshold; the larger one wins so that a diagonal glance reports the
    dominant axis rather than flickering between the two.
    """
    user_h = h if mirrored else 1.0 - h
    dh = user_h - 0.5
    dv = v - thresholds.vertical_center

    def exceed(value: float, limit: float, direction: str) -> float:
        # Values already in `direction` only need to stay above the lowered
        # hysteresis bound to remain there.
        eff = limit * thresholds.hysteresis if current == direction else limit
        return value / eff if eff > 0 else 0.0

    candidates = {
        LEFT: exceed(-dh, thresholds.horizontal, LEFT),
        RIGHT: exceed(dh, thresholds.horizontal, RIGHT),
        UP: exceed(-dv, thresholds.up, UP),
        DOWN: exceed(dv, thresholds.down, DOWN),
    }
    best = max(candidates, key=candidates.get)
    if candidates[best] >= 1.0:
        return best
    return CENTER


@dataclass
class GazeState:
    direction: str
    h: float
    v: float
    raw_h: float
    raw_v: float


class GazeClassifier:
    """Stateful wrapper: exponential smoothing + hysteresis over ``classify``."""

    def __init__(
        self,
        thresholds: GazeThresholds = GazeThresholds(),
        smoothing: float = 0.35,
        mirrored: bool = True,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.thresholds = thresholds
        self.smoothing = smoothing
        self.mirrored = mirrored
        self.direction = CENTER
        self._h: Optional[float] = None
        self._v: Optional[float] = None

    def reset(self) -> None:
        self.direction = CENTER
        self._h = None
        self._v = None

    def update(self, measure: GazeMeasure, eyes_closed: bool = False) -> GazeState:
        """Feed one frame. While eyes are closed the previous state is held."""
        if self._h is None or self._v is None:
            self._h, self._v = measure.h, measure.v
        elif not eyes_closed:
            a = self.smoothing
            self._h = a * measure.h + (1 - a) * self._h
            self._v = a * measure.v + (1 - a) * self._v

        if not eyes_closed:
            self.direction = classify(
                self._h, self._v, self.thresholds, self.mirrored, current=self.direction
            )
        return GazeState(
            direction=self.direction,
            h=self._h,
            v=self._v,
            raw_h=measure.h,
            raw_v=measure.v,
        )
