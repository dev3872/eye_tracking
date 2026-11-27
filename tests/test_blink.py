import numpy as np
import pytest

from eye_tracker import blink as b
from eye_tracker.landmarks import LEFT_EYE_EAR, NUM_LANDMARKS, RIGHT_EYE_EAR


def eye_points(width: float = 40.0, opening: float = 12.0) -> np.ndarray:
    """Six EAR points for a symmetric eye: corners at x=0 and x=width."""
    return np.array(
        [
            [0.0, 0.0],                     # p1 outer corner
            [width * 0.3, -opening / 2],    # p2 upper
            [width * 0.7, -opening / 2],    # p3 upper
            [width, 0.0],                   # p4 inner corner
            [width * 0.7, opening / 2],     # p5 lower
            [width * 0.3, opening / 2],     # p6 lower
        ]
    )


def face_with_opening(opening: float) -> np.ndarray:
    lm = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    lm[list(RIGHT_EYE_EAR)] = eye_points(opening=opening) + [100, 100]
    lm[list(LEFT_EYE_EAR)] = eye_points(opening=opening) + [200, 100]
    return lm


def test_ear_open_eye():
    # EAR = (12 + 12) / (2 * 40) = 0.3
    assert b.eye_aspect_ratio(eye_points(40, 12)) == pytest.approx(0.3)


def test_ear_closed_eye_is_small():
    assert b.eye_aspect_ratio(eye_points(40, 2)) == pytest.approx(0.05)


def test_ear_scale_invariant():
    assert b.eye_aspect_ratio(eye_points(40, 12)) == pytest.approx(b.eye_aspect_ratio(eye_points(80, 24)))


def test_ear_degenerate_width():
    assert b.eye_aspect_ratio(eye_points(0.0, 12)) == 0.0


def test_ear_shape_check():
    with pytest.raises(ValueError):
        b.eye_aspect_ratio(np.zeros((5, 2)))


def test_both_eyes_from_landmarks():
    right, left = b.both_eyes_ear(face_with_opening(12))
    assert right == pytest.approx(0.3)
    assert left == pytest.approx(0.3)


def test_blink_detector_counts_after_reopen():
    d = b.BlinkDetector(threshold=0.21, min_frames=2)
    assert d.update(0.30) is None
    assert d.update(0.15) is None          # closed frame 1
    assert d.is_closed
    assert d.update(0.10) is None          # closed frame 2
    event = d.update(0.28)                 # reopened
    assert event is not None
    assert event.count == 1
    assert event.duration_frames == 2
    assert not d.is_closed
    assert d.count == 1


def test_blink_detector_ignores_single_frame_dip():
    d = b.BlinkDetector(threshold=0.21, min_frames=2)
    d.update(0.30)
    d.update(0.10)
    assert d.update(0.30) is None
    assert d.count == 0


def test_blink_detector_from_landmarks():
    d = b.BlinkDetector(threshold=0.21, min_frames=1)
    assert d.update_from_landmarks(face_with_opening(12)) is None
    assert d.update_from_landmarks(face_with_opening(2)) is None
    event = d.update_from_landmarks(face_with_opening(12))
    assert event is not None and event.count == 1


def test_blink_detector_reopen_hysteresis():
    d = b.BlinkDetector(threshold=0.21, min_frames=1, reopen_margin=0.03)
    d.update(0.30)
    d.update(0.18)                          # closed
    assert d.update(0.22) is None           # above threshold but inside the margin: still closed
    assert d.is_closed
    event = d.update(0.25)                  # clearly open
    assert event is not None and event.count == 1 and event.duration_frames == 2


def test_blink_detector_long_closure_is_not_a_blink():
    d = b.BlinkDetector(threshold=0.21, min_frames=2, max_frames=5)
    for _ in range(10):
        d.update(0.1)
    event = d.update(0.3)
    assert event is not None
    assert event.is_blink is False
    assert event.duration_frames == 10
    assert d.count == 0


def test_blink_detector_reset_drops_partial_closure():
    d = b.BlinkDetector(threshold=0.21, min_frames=2)
    d.update(0.1)
    d.update(0.1)
    d.reset()
    assert d.update(0.3) is None
    assert d.count == 0
