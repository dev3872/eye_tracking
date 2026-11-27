import numpy as np
import pytest

from eye_tracker import gaze as g
from eye_tracker.landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_IRIS_CENTER,
    NUM_LANDMARKS,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS_CENTER,
)


def synthetic_face(h: float = 0.5, v: float = 0.0, eye_width: float = 40.0, tilt_deg: float = 0.0) -> np.ndarray:
    """Build a landmark array where both irises sit at the given (h, v) ratios.

    Eyes are placed around (200, 150) and (300, 150); ``tilt_deg`` rotates the
    whole face about the midpoint to check that head roll is handled.
    """
    lm = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    centre = np.array([250.0, 150.0])
    for (outer, inner), iris_idx, cx in (
        (RIGHT_EYE_CORNERS, RIGHT_IRIS_CENTER, 200.0),   # subject's right eye: image-left
        (LEFT_EYE_CORNERS, LEFT_IRIS_CENTER, 300.0),     # subject's left eye: image-right
    ):
        left_corner = np.array([cx - eye_width / 2, 150.0])
        right_corner = np.array([cx + eye_width / 2, 150.0])
        iris = left_corner + np.array([h * eye_width, v * eye_width])
        # Right eye: outer corner is on the image-left; left eye: outer is image-right.
        if cx < 250:
            lm[outer], lm[inner] = left_corner, right_corner
        else:
            lm[outer], lm[inner] = right_corner, left_corner
        lm[iris_idx] = iris

    if tilt_deg:
        th = np.deg2rad(tilt_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        lm = ((lm - centre) @ rot.T + centre).astype(np.float32)
    return lm


def test_measure_centre():
    m = g.measure_gaze(synthetic_face(0.5, 0.0))
    assert m is not None
    assert m.h == pytest.approx(0.5, abs=1e-6)
    assert m.v == pytest.approx(0.0, abs=1e-6)
    assert m.eye_width == pytest.approx(40.0)


@pytest.mark.parametrize("h,v", [(0.3, 0.0), (0.7, 0.0), (0.5, -0.1), (0.5, 0.08), (0.62, -0.05)])
def test_measure_roundtrip(h, v):
    m = g.measure_gaze(synthetic_face(h, v))
    assert m.h == pytest.approx(h, abs=1e-5)
    assert m.v == pytest.approx(v, abs=1e-5)


def test_measure_is_invariant_to_head_roll():
    flat = g.measure_gaze(synthetic_face(0.35, 0.05))
    tilted = g.measure_gaze(synthetic_face(0.35, 0.05, tilt_deg=20))
    assert tilted.h == pytest.approx(flat.h, abs=1e-4)
    assert tilted.v == pytest.approx(flat.v, abs=1e-4)


def test_measure_scales_with_distance():
    near = g.measure_gaze(synthetic_face(0.4, 0.03, eye_width=80))
    far = g.measure_gaze(synthetic_face(0.4, 0.03, eye_width=20))
    assert near.h == pytest.approx(far.h, abs=1e-5)
    assert near.v == pytest.approx(far.v, abs=1e-5)


def test_degenerate_eye_returns_none():
    lm = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    assert g.measure_gaze(lm) is None


def test_classify_centre_and_sides_mirrored():
    t = g.GazeThresholds(vertical_center=0.0)
    assert g.classify(0.5, 0.0, t, mirrored=True) == g.CENTER
    # Mirrored frame: iris toward image-left means the user looked left.
    assert g.classify(0.5 - t.horizontal - 0.01, 0.0, t, mirrored=True) == g.LEFT
    assert g.classify(0.5 + t.horizontal + 0.01, 0.0, t, mirrored=True) == g.RIGHT


def test_classify_unmirrored_flips_horizontal():
    t = g.GazeThresholds(vertical_center=0.0)
    assert g.classify(0.5 - t.horizontal - 0.01, 0.0, t, mirrored=False) == g.RIGHT
    assert g.classify(0.5 + t.horizontal + 0.01, 0.0, t, mirrored=False) == g.LEFT


def test_classify_vertical():
    t = g.GazeThresholds(vertical_center=0.0)
    assert g.classify(0.5, -(t.up + 0.01), t) == g.UP
    assert g.classify(0.5, t.down + 0.01, t) == g.DOWN
    assert g.classify(0.5, -(t.up - 0.01), t) == g.CENTER


def test_default_neutral_offset_is_centre():
    t = g.GazeThresholds()
    assert t.vertical_center < 0
    assert g.classify(0.5, t.vertical_center, t) == g.CENTER
    assert g.classify(0.5, 0.0, t) == g.DOWN


def test_edge_on_eye_is_discarded():
    lm = synthetic_face(0.5, 0.0)
    # Push the right iris far outside its corners; the left eye should carry the result.
    lm[RIGHT_IRIS_CENTER] = lm[RIGHT_EYE_CORNERS[0]] + np.array([-100.0, 0.0], dtype=np.float32)
    m = g.measure_gaze(lm)
    assert m is not None
    assert m.h == pytest.approx(0.5, abs=1e-6)


def test_classify_vertical_center_offset():
    t = g.GazeThresholds(vertical_center=-0.03)
    assert g.classify(0.5, -0.03, t) == g.CENTER
    assert g.classify(0.5, -0.03 - t.up - 0.01, t) == g.UP


def test_classify_dominant_axis_wins():
    t = g.GazeThresholds(horizontal=0.1, up=0.1, down=0.1, vertical_center=0.0)
    # Horizontal exceeds threshold by 2x, vertical by 1.2x.
    assert g.classify(0.5 - 0.2, -0.12, t, mirrored=True) == g.LEFT
    assert g.classify(0.5 - 0.12, -0.2, t, mirrored=True) == g.UP


def test_classify_hysteresis_holds_direction():
    t = g.GazeThresholds(horizontal=0.1, hysteresis=0.5, vertical_center=0.0)
    just_inside = 0.5 - 0.07  # inside the enter threshold, outside the exit threshold
    assert g.classify(just_inside, 0.0, t, current=g.CENTER) == g.CENTER
    assert g.classify(just_inside, 0.0, t, current=g.LEFT) == g.LEFT
    assert g.classify(0.5 - 0.04, 0.0, t, current=g.LEFT) == g.CENTER


def test_classifier_smooths_and_holds_during_blink():
    c = g.GazeClassifier(g.GazeThresholds(horizontal=0.1, vertical_center=0.0), smoothing=0.5, mirrored=True)
    centre = g.measure_gaze(synthetic_face(0.5, 0.0))
    left = g.measure_gaze(synthetic_face(0.3, 0.0))

    s = c.update(centre)
    assert s.direction == g.CENTER
    # First frame of a glance left is smoothed: 0.5*0.3 + 0.5*0.5 = 0.4 -> CENTER
    s = c.update(left)
    assert s.h == pytest.approx(0.4)
    assert s.direction == g.CENTER
    s = c.update(left)
    assert s.h == pytest.approx(0.35)
    assert s.direction == g.LEFT

    # Eyes closed: nothing moves even if the iris estimate jumps.
    s = c.update(centre, eyes_closed=True)
    assert s.h == pytest.approx(0.35)
    assert s.direction == g.LEFT

    c.reset()
    assert c.direction == g.CENTER
    assert c.update(left).h == pytest.approx(0.3)


def test_classifier_rejects_bad_smoothing():
    with pytest.raises(ValueError):
        g.GazeClassifier(smoothing=0.0)
