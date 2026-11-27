"""Hands-free mouse: gaze drives a cursor velocity, deliberate blinks click.

The control scheme is a *joystick*, not a pointer: looking away from centre
pushes the cursor in that direction at a speed that grows with how far the
iris travels, and looking back to centre stops it. This needs no calibration,
which keeps it consistent with the rest of the tracker.

Blink gestures (natural short blinks are ignored):

* one deliberate blink (eyes shut ``click_min_s``-``click_max_s``) = left click
* two deliberate blinks, the second starting within ``double_window_s`` of the
  first reopening = right click
* eyes held shut longer than ``click_max_s`` = cancel, no click

The cursor is frozen from the moment the eyes start closing until the click
decision is made plus a short grace period, and the click lands on the
position buffered when the eyes started to close, so gaze drift during the
blink cannot drag the cursor off its target.

Everything in here is pure Python state; the OS is only touched through a
``MouseBackend``. ``NullBackend`` records calls for tests, ``DryRunBackend``
logs them, ``PynputBackend`` moves the real cursor.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .gaze import GazeThresholds

LEFT_BUTTON = "left"
RIGHT_BUTTON = "right"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class MouseBackend(Protocol):
    def screen_size(self) -> tuple[int, int]: ...

    def position(self) -> tuple[int, int]: ...

    def move_to(self, x: int, y: int) -> None: ...

    def click(self, button: str, x: int, y: int) -> None: ...


class NullBackend:
    """In-memory cursor: keeps its own position and records every call."""

    def __init__(self, screen: tuple[int, int] = (1920, 1080),
                 position: Optional[tuple[int, int]] = None) -> None:
        self._screen = screen
        self._pos = position if position is not None else (screen[0] // 2, screen[1] // 2)
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[tuple[str, int, int]] = []

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def position(self) -> tuple[int, int]:
        return self._pos

    def move_to(self, x: int, y: int) -> None:
        self._pos = (x, y)
        self.moves.append((x, y))

    def click(self, button: str, x: int, y: int) -> None:
        self._pos = (x, y)
        self.clicks.append((button, x, y))


class DryRunBackend(NullBackend):
    """Like ``NullBackend`` but narrates what it would do. Moves are rate
    limited so the log stays readable; clicks are always reported."""

    def __init__(self, screen: tuple[int, int], emit: Callable[[str], None],
                 move_log_interval_s: float = 0.25) -> None:
        super().__init__(screen=screen)
        self._emit = emit
        self._interval = move_log_interval_s
        self._last_move_log = -math.inf

    def move_to(self, x: int, y: int) -> None:
        super().move_to(x, y)
        now = time.monotonic()
        if now - self._last_move_log >= self._interval:
            self._last_move_log = now
            self._emit(f"dry-run: move -> ({x}, {y})")

    def click(self, button: str, x: int, y: int) -> None:
        super().click(button, x, y)
        self._emit(f"dry-run: click {button} at ({x}, {y})")


def detect_screen_size() -> Optional[tuple[int, int]]:
    """Primary display size in pixels, or None if it cannot be determined.

    Uses the libraries pynput already pulls in per platform (Quartz on macOS,
    Xlib on Linux) and ctypes on Windows, with tkinter as a last resort.
    """
    try:
        if sys.platform == "darwin":
            import Quartz  # type: ignore[import-not-found]

            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return int(bounds.size.width), int(bounds.size.height)
        if sys.platform.startswith("win"):
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        from Xlib import display  # type: ignore[import-not-found]

        screen = display.Display().screen()
        return int(screen.width_in_pixels), int(screen.height_in_pixels)
    except Exception:
        pass
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return None


class PynputBackend:
    """Real cursor via pynput. Imported lazily so the tracker still runs where
    there is no display server or input permission."""

    def __init__(self, screen: Optional[tuple[int, int]] = None) -> None:
        from pynput.mouse import Button, Controller

        self._buttons = {LEFT_BUTTON: Button.left, RIGHT_BUTTON: Button.right}
        self._mouse = Controller()
        detected = screen or detect_screen_size()
        if detected is None:
            raise RuntimeError("could not determine the screen size; pass --screen WxH")
        self._screen = detected

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def position(self) -> tuple[int, int]:
        x, y = self._mouse.position
        return int(x), int(y)

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def click(self, button: str, x: int, y: int) -> None:
        self._mouse.position = (x, y)
        self._mouse.click(self._buttons[button], 1)


# --------------------------------------------------------------------------- #
# Pure control logic
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MouseConfig:
    """Joystick mapping and click timing.

    Dead zones default to the gaze thresholds so the cursor only moves once
    the HUD would also report LEFT/RIGHT/UP/DOWN. ``full_speed_h`` /
    ``full_speed_v`` are the iris excursions (same units as ``h`` / ``v``) at
    which ``max_speed`` is reached; between the dead zone and there the speed
    follows ``excursion ** gain``.
    """

    max_speed: float = 900.0
    gain: float = 2.0
    dead_zone_h: float = GazeThresholds.horizontal
    dead_zone_up: float = GazeThresholds.up
    dead_zone_down: float = GazeThresholds.down
    vertical_center: float = GazeThresholds.vertical_center
    full_speed_h: float = 0.22
    full_speed_v: float = 0.15
    click_min_s: float = 0.35
    click_max_s: float = 1.0
    double_window_s: float = 0.7
    click_freeze_s: float = 0.3
    face_lost_release_s: float = 1.0
    edge_margin: int = 2
    max_dt: float = 0.1


def _axis_speed(excursion: float, dead_zone: float, full: float, max_speed: float, gain: float) -> float:
    """Signed speed for one axis from a signed excursion past the dead zone."""
    magnitude = abs(excursion)
    if magnitude <= dead_zone:
        return 0.0
    span = max(full - dead_zone, 1e-6)
    norm = min((magnitude - dead_zone) / span, 1.0)
    return math.copysign(max_speed * norm ** gain, excursion)


def gaze_velocity(h: float, v: float, config: MouseConfig = MouseConfig(),
                  mirrored: bool = True) -> tuple[float, float]:
    """(vx, vy) in px/s, screen convention: +x right, +y down.

    Directions follow the user's point of view exactly as ``gaze.classify``
    does: looking to their own left moves the cursor left, looking up moves
    it up (decreasing y).
    """
    user_h = h if mirrored else 1.0 - h
    dh = user_h - 0.5
    dv = v - config.vertical_center
    vx = _axis_speed(dh, config.dead_zone_h, config.full_speed_h, config.max_speed, config.gain)
    dead_v = config.dead_zone_up if dv < 0 else config.dead_zone_down
    vy = _axis_speed(dv, dead_v, config.full_speed_v, config.max_speed, config.gain)
    return vx, vy


def clamp_to_screen(x: float, y: float, screen: tuple[int, int], margin: int = 0) -> tuple[float, float]:
    w, h = screen
    return (
        min(max(x, float(margin)), float(w - 1 - margin)),
        min(max(y, float(margin)), float(h - 1 - margin)),
    )


@dataclass(frozen=True)
class ClickEvent:
    button: str
    t: float


class BlinkClicker:
    """Turns an eyes-closed signal into click gestures by closure duration.

    Feed ``update(closed, t)`` every frame with ``t`` in seconds (camera
    timestamps). A left click is reported only once ``double_window_s`` has
    passed without a second deliberate blink starting, so a right click never
    fires a stray left click first.
    """

    def __init__(self, min_s: float = 0.35, max_s: float = 1.0, double_window_s: float = 0.7) -> None:
        if not 0 < min_s < max_s:
            raise ValueError("need 0 < min_s < max_s")
        self.min_s = min_s
        self.max_s = max_s
        self.double_window_s = double_window_s
        self._closed_since: Optional[float] = None
        self._pending_since: Optional[float] = None

    @property
    def closed(self) -> bool:
        return self._closed_since is not None

    @property
    def pending(self) -> bool:
        """A deliberate blink has been seen and is waiting for a possible second one."""
        return self._pending_since is not None

    def reset(self) -> None:
        self._closed_since = None
        self._pending_since = None

    def abort_closure(self) -> None:
        """Drop the closure in progress without classifying it (face lost)."""
        self._closed_since = None

    def update(self, closed: bool, t: float) -> Optional[ClickEvent]:
        if closed:
            if self._closed_since is None:
                if self._pending_since is not None and t - self._pending_since > self.double_window_s:
                    self._pending_since = None
                    self._closed_since = t
                    return ClickEvent(LEFT_BUTTON, t)
                self._closed_since = t
            return None

        if self._closed_since is not None:
            duration = t - self._closed_since
            self._closed_since = None
            if duration > self.max_s:
                self._pending_since = None
                return None
            if duration >= self.min_s:
                if self._pending_since is not None:
                    self._pending_since = None
                    return ClickEvent(RIGHT_BUTTON, t)
                self._pending_since = t
                return None

        if self._pending_since is not None and t - self._pending_since >= self.double_window_s:
            self._pending_since = None
            return ClickEvent(LEFT_BUTTON, t)
        return None


@dataclass
class MouseUpdate:
    velocity: tuple[float, float] = (0.0, 0.0)
    position: Optional[tuple[int, int]] = None
    frozen: bool = False
    released: bool = False
    click: Optional[ClickEvent] = None
    released_changed: bool = False


class GazeMouseController:
    """Stateful wrapper: integrates ``gaze_velocity`` into cursor moves and
    routes ``BlinkClicker`` events to the backend.

    Call ``update`` once per frame with the *smoothed* gaze, whether the eyes
    are closed, and the frame time in seconds. Pass ``h=None, v=None`` when no
    face is visible: the cursor stops at once, and after
    ``face_lost_release_s`` control is released (pending gestures dropped)
    until the face is back.
    """

    def __init__(self, backend: MouseBackend, config: MouseConfig = MouseConfig(),
                 mirrored: bool = True) -> None:
        self.backend = backend
        self.config = config
        self.mirrored = mirrored
        self.enabled = True
        self.clicker = BlinkClicker(config.click_min_s, config.click_max_s, config.double_window_s)
        self.last_click: Optional[ClickEvent] = None
        self.released = False
        self._last_t: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._freeze_until = -math.inf
        self._anchor: Optional[tuple[int, int]] = None
        self._was_closed = False
        self._rem_x = 0.0
        self._rem_y = 0.0

    def reset(self) -> None:
        self.clicker.reset()
        self.released = False
        self._last_t = None
        self._lost_since = None
        self._freeze_until = -math.inf
        self._anchor = None
        self._was_closed = False
        self._rem_x = self._rem_y = 0.0

    def _dt(self, t: float) -> float:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        return min(dt, self.config.max_dt)

    def update(self, h: Optional[float], v: Optional[float], eyes_closed: bool, t: float) -> MouseUpdate:
        dt = self._dt(t)
        out = MouseUpdate(position=self.backend.position(), released=self.released)

        if h is None or v is None:
            self.clicker.abort_closure()
            self._was_closed = False
            if self._lost_since is None:
                self._lost_since = t
            elif not self.released and t - self._lost_since >= self.config.face_lost_release_s:
                self.released = True
                self.clicker.reset()
                self._anchor = None
                out.released = True
                out.released_changed = True
            return out

        self._lost_since = None
        if self.released:
            self.released = False
            out.released = False
            out.released_changed = True

        if not self.enabled:
            self.clicker.reset()
            self._anchor = None
            self._was_closed = False
            return out

        if eyes_closed and not self._was_closed:
            self._anchor = self.backend.position()
        if self._was_closed and not eyes_closed:
            self._freeze_until = max(self._freeze_until, t + self.config.click_freeze_s)
        self._was_closed = eyes_closed

        click = self.clicker.update(eyes_closed, t)
        if click is not None:
            x, y = self._anchor if self._anchor is not None else self.backend.position()
            self.backend.click(click.button, x, y)
            self.last_click = click
            out.click = click
            out.position = (x, y)
            self._freeze_until = max(self._freeze_until, t + self.config.click_freeze_s)

        frozen = eyes_closed or self.clicker.pending or t < self._freeze_until
        out.frozen = frozen
        if frozen:
            self._rem_x = self._rem_y = 0.0
            return out
        self._anchor = None

        vx, vy = gaze_velocity(h, v, self.config, self.mirrored)
        out.velocity = (vx, vy)
        if dt <= 0 or (vx == 0.0 and vy == 0.0):
            self._rem_x = self._rem_y = 0.0
            return out

        px, py = self.backend.position()
        nx = px + vx * dt + self._rem_x
        ny = py + vy * dt + self._rem_y
        nx, ny = clamp_to_screen(nx, ny, self.backend.screen_size(), self.config.edge_margin)
        ix, iy = int(round(nx)), int(round(ny))
        self._rem_x = nx - ix
        self._rem_y = ny - iy
        if (ix, iy) != (px, py):
            self.backend.move_to(ix, iy)
        out.position = (ix, iy)
        return out
