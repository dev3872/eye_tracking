import pytest

from eye_tracker import mouse as m
from eye_tracker.gaze import GazeThresholds

T = GazeThresholds()
CENTER_V = T.vertical_center
CFG = m.MouseConfig(max_speed=1000.0, gain=2.0)
FPS = 30.0
DT = 1.0 / FPS


def run_frames(ctrl: m.GazeMouseController, frames, t0: float = 0.0) -> list[m.MouseUpdate]:
    """Feed (h, v, closed) tuples at FPS; h=None means no face."""
    out = []
    for i, (h, v, closed) in enumerate(frames):
        out.append(ctrl.update(h, v, closed, t0 + i * DT))
    return out


def closed_frames(n: int, h: float = 0.5, v: float = CENTER_V):
    return [(h, v, True)] * n


def open_frames(n: int, h: float = 0.5, v: float = CENTER_V):
    return [(h, v, False)] * n


def seconds(s: float) -> int:
    return int(round(s * FPS))


# ---------------------------------------------------------------- velocity --


def test_dead_zone_gives_zero_velocity():
    assert m.gaze_velocity(0.5, CENTER_V, CFG) == (0.0, 0.0)
    assert m.gaze_velocity(0.5 + T.horizontal * 0.99, CENTER_V, CFG) == (0.0, 0.0)
    assert m.gaze_velocity(0.5, CENTER_V - T.up * 0.99, CFG) == (0.0, 0.0)
    assert m.gaze_velocity(0.5, CENTER_V + T.down * 0.99, CFG) == (0.0, 0.0)


def test_mirrored_left_moves_cursor_left():
    vx, vy = m.gaze_velocity(0.3, CENTER_V, CFG, mirrored=True)
    assert vx < 0 and vy == 0.0


def test_mirrored_right_moves_cursor_right():
    vx, _ = m.gaze_velocity(0.7, CENTER_V, CFG, mirrored=True)
    assert vx > 0


def test_unmirrored_flips_horizontal():
    assert m.gaze_velocity(0.3, CENTER_V, CFG, mirrored=False)[0] > 0
    assert m.gaze_velocity(0.7, CENTER_V, CFG, mirrored=False)[0] < 0


def test_up_decreases_y_and_down_increases_y():
    _, vy_up = m.gaze_velocity(0.5, CENTER_V - 0.15, CFG)
    _, vy_down = m.gaze_velocity(0.5, CENTER_V + 0.15, CFG)
    assert vy_up < 0
    assert vy_down > 0


def test_speed_is_monotonic_in_excursion_and_capped():
    speeds = [abs(m.gaze_velocity(0.5 - d, CENTER_V, CFG)[0]) for d in (0.09, 0.12, 0.16, 0.2, 0.22, 0.3)]
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))
    assert speeds[0] > 0
    assert speeds[-1] == pytest.approx(CFG.max_speed)
    assert speeds[-2] == pytest.approx(CFG.max_speed)


def test_gain_makes_small_glances_gentler():
    linear = m.MouseConfig(max_speed=1000.0, gain=1.0)
    squared = m.MouseConfig(max_speed=1000.0, gain=2.0)
    h = 0.5 - (T.horizontal + 0.5 * (CFG.full_speed_h - T.horizontal))  # halfway to full speed
    v_lin = abs(m.gaze_velocity(h, CENTER_V, linear)[0])
    v_sq = abs(m.gaze_velocity(h, CENTER_V, squared)[0])
    assert v_lin == pytest.approx(500.0)
    assert v_sq == pytest.approx(250.0)


def test_clamp_to_screen():
    assert m.clamp_to_screen(-50, -50, (1920, 1080)) == (0.0, 0.0)
    assert m.clamp_to_screen(5000, 5000, (1920, 1080)) == (1919.0, 1079.0)
    assert m.clamp_to_screen(5000, 500, (1920, 1080), margin=2) == (1917.0, 500.0)


# -------------------------------------------------------------- controller --


def test_controller_moves_left_and_stops_in_dead_zone():
    be = m.NullBackend(screen=(1920, 1080), position=(960, 540))
    ctrl = m.GazeMouseController(be, CFG)
    run_frames(ctrl, open_frames(seconds(1.0), h=0.3))
    assert be.position()[0] < 960
    assert be.position()[1] == 540
    x_after_move = be.position()[0]
    run_frames(ctrl, open_frames(seconds(1.0)), t0=2.0)
    assert be.position()[0] == x_after_move


def test_controller_up_decreases_y():
    be = m.NullBackend(screen=(1920, 1080), position=(960, 540))
    ctrl = m.GazeMouseController(be, CFG)
    run_frames(ctrl, open_frames(seconds(1.0), v=CENTER_V - 0.15))
    assert be.position()[1] < 540


def test_controller_clamps_at_screen_edge():
    be = m.NullBackend(screen=(800, 600), position=(790, 300))
    ctrl = m.GazeMouseController(be, m.MouseConfig(max_speed=5000.0, edge_margin=2))
    run_frames(ctrl, open_frames(seconds(2.0), h=0.9))
    assert be.position() == (797, 300)
    assert all(0 <= x <= 799 and 0 <= y <= 599 for x, y in be.moves)


def test_controller_full_speed_covers_expected_distance():
    be = m.NullBackend(screen=(10000, 1000), position=(5000, 500))
    ctrl = m.GazeMouseController(be, CFG)
    run_frames(ctrl, open_frames(seconds(1.0) + 1, h=0.9))
    assert be.position()[0] == pytest.approx(5000 + CFG.max_speed, abs=2)


def test_controller_stops_when_eyes_closed_and_face_lost():
    be = m.NullBackend(position=(960, 540))
    ctrl = m.GazeMouseController(be, CFG)
    run_frames(ctrl, closed_frames(10, h=0.2))
    assert be.moves == []
    run_frames(ctrl, [(None, None, False)] * 10, t0=1.0)
    assert be.moves == []


def test_controller_releases_after_face_lost_one_second():
    be = m.NullBackend(position=(960, 540))
    ctrl = m.GazeMouseController(be, m.MouseConfig(face_lost_release_s=1.0))
    run_frames(ctrl, closed_frames(seconds(0.5)))                 # deliberate blink in progress
    ups = run_frames(ctrl, [(None, None, False)] * seconds(1.5), t0=0.5)
    assert any(u.released_changed and u.released for u in ups)
    assert ctrl.released
    assert not ctrl.clicker.pending and not ctrl.clicker.closed
    back = ctrl.update(0.5, CENTER_V, False, 3.0)
    assert back.released_changed and not back.released
    assert be.clicks == []


def test_controller_paused_does_nothing():
    be = m.NullBackend(position=(960, 540))
    ctrl = m.GazeMouseController(be, CFG)
    ctrl.enabled = False
    run_frames(ctrl, open_frames(seconds(1.0), h=0.2))
    run_frames(ctrl, closed_frames(seconds(0.5)), t0=2.0)
    run_frames(ctrl, open_frames(seconds(1.5)), t0=3.0)
    assert be.moves == [] and be.clicks == []


def test_controller_respects_external_mouse_position():
    be = m.NullBackend(screen=(1920, 1080), position=(100, 100))
    ctrl = m.GazeMouseController(be, CFG)
    run_frames(ctrl, open_frames(5, h=0.3))
    be._pos = (1500, 700)
    run_frames(ctrl, open_frames(5, h=0.3), t0=1.0)
    assert 1300 < be.position()[0] < 1500 and be.position()[1] == 700


# ------------------------------------------------------------ blink clicks --


def feed(clicker: m.BlinkClicker, pattern, t0: float = 0.0) -> list[m.ClickEvent]:
    events = []
    for i, closed in enumerate(pattern):
        ev = clicker.update(closed, t0 + i * DT)
        if ev is not None:
            events.append(ev)
    return events


def test_short_natural_blink_does_not_click():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    events = feed(c, [False] * 5 + [True] * seconds(0.15) + [False] * seconds(2.0))
    assert events == []


def test_long_blink_is_left_click_after_double_window():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    events = feed(c, [False] * 5 + [True] * seconds(0.5) + [False] * seconds(2.0))
    assert [e.button for e in events] == [m.LEFT_BUTTON]
    reopen_t = (5 + seconds(0.5)) * DT
    assert events[0].t == pytest.approx(reopen_t + 0.7, abs=DT)


def test_left_click_waits_for_window_before_firing():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    assert feed(c, [True] * seconds(0.5) + [False] * seconds(0.5)) == []
    assert c.pending


def test_two_long_blinks_are_right_click():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    pattern = [True] * seconds(0.5) + [False] * seconds(0.3) + [True] * seconds(0.5) + [False] * seconds(2.0)
    events = feed(c, pattern)
    assert [e.button for e in events] == [m.RIGHT_BUTTON]


def test_second_blink_after_window_is_two_left_clicks():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    pattern = [True] * seconds(0.5) + [False] * seconds(1.0) + [True] * seconds(0.5) + [False] * seconds(2.0)
    events = feed(c, pattern)
    assert [e.button for e in events] == [m.LEFT_BUTTON, m.LEFT_BUTTON]


def test_overlong_closure_does_not_click():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    events = feed(c, [True] * seconds(1.5) + [False] * seconds(2.0))
    assert events == []
    assert not c.pending


def test_overlong_closure_cancels_pending_click():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    pattern = [True] * seconds(0.5) + [False] * seconds(0.2) + [True] * seconds(1.5) + [False] * seconds(2.0)
    assert feed(c, pattern) == []


def test_natural_blink_between_does_not_disturb_pending_left_click():
    c = m.BlinkClicker(min_s=0.35, max_s=1.0, double_window_s=0.7)
    pattern = [True] * seconds(0.5) + [False] * seconds(0.2) + [True] * 3 + [False] * seconds(2.0)
    assert [e.button for e in feed(c, pattern)] == [m.LEFT_BUTTON]


def test_clicker_rejects_bad_bounds():
    with pytest.raises(ValueError):
        m.BlinkClicker(min_s=1.0, max_s=0.5)


def test_controller_clicks_at_position_buffered_when_eyes_closed():
    be = m.NullBackend(screen=(1920, 1080), position=(960, 540))
    cfg = m.MouseConfig(max_speed=1000.0, click_freeze_s=0.3, double_window_s=0.7)
    ctrl = m.GazeMouseController(be, cfg)
    run_frames(ctrl, open_frames(seconds(0.5), h=0.3))
    anchor = be.position()
    # Gaze wanders far right while the eyes are shut and during the hold afterwards.
    n_closed = seconds(0.5)
    ups = run_frames(ctrl, closed_frames(n_closed, h=0.9), t0=0.5)
    assert all(u.frozen for u in ups)
    ups = run_frames(ctrl, open_frames(seconds(1.5), h=0.9), t0=0.5 + n_closed * DT)
    assert be.clicks == [(m.LEFT_BUTTON, anchor[0], anchor[1])]
    click_idx = next(i for i, u in enumerate(ups) if u.click is not None)
    assert all(u.frozen for u in ups[: click_idx + 1])
    # Movement resumes after the grace period, in the direction of the new gaze.
    assert not ups[-1].frozen
    assert be.position()[0] > anchor[0]


def test_controller_right_click_uses_first_closure_anchor():
    be = m.NullBackend(screen=(1920, 1080), position=(400, 400))
    ctrl = m.GazeMouseController(be, m.MouseConfig(max_speed=1000.0))
    pattern = (
        closed_frames(seconds(0.5), h=0.8)
        + open_frames(seconds(0.3), h=0.8)
        + closed_frames(seconds(0.5), h=0.8)
        + open_frames(seconds(0.2), h=0.8)
    )
    run_frames(ctrl, pattern)
    assert be.clicks == [(m.RIGHT_BUTTON, 400, 400)]
    assert be.moves == []


def test_dry_run_backend_logs_clicks_and_moves():
    lines: list[str] = []
    be = m.DryRunBackend(screen=(1920, 1080), emit=lines.append, move_log_interval_s=0.0)
    be.move_to(10, 20)
    be.click(m.LEFT_BUTTON, 10, 20)
    assert lines == ["dry-run: move -> (10, 20)", "dry-run: click left at (10, 20)"]
    assert be.position() == (10, 20)
