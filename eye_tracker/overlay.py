"""All OpenCV drawing for the tracker window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import gaze as g
from .landmarks import (
    LEFT_EYE_CONTOUR,
    LEFT_IRIS,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS,
    RIGHT_IRIS_CENTER,
)

# BGR colours
EYE_COLOR = (90, 220, 120)
IRIS_COLOR = (60, 160, 255)
IRIS_CENTER_COLOR = (0, 90, 255)
TEXT_COLOR = (240, 240, 240)
DIM_COLOR = (150, 150, 150)
ACCENT = (255, 190, 60)
ALERT = (60, 60, 230)
PANEL = (20, 20, 20)

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class MouseHud:
    enabled: bool
    velocity: tuple[float, float] = (0.0, 0.0)
    frozen: bool = False
    released: bool = False
    click_flash: Optional[str] = None
    dry_run: bool = False


@dataclass
class HudState:
    direction: str
    h: Optional[float]
    v: Optional[float]
    ear: Optional[float]
    blink_count: int
    eyes_closed: bool
    fps: float
    face_found: bool
    show_landmarks: bool = True
    mirrored: bool = True
    vertical_center: float = g.GazeThresholds.vertical_center
    mouse: Optional[MouseHud] = None


def _put_text(img, text, org, scale=0.6, color=TEXT_COLOR, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_eyes(img: np.ndarray, landmarks: np.ndarray) -> None:
    for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
        pts = landmarks[contour].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=EYE_COLOR, thickness=1, lineType=cv2.LINE_AA)

    for iris, center_idx in ((RIGHT_IRIS, RIGHT_IRIS_CENTER), (LEFT_IRIS, LEFT_IRIS_CENTER)):
        ring = landmarks[iris[1:]]
        center = landmarks[center_idx]
        radius = float(np.mean(np.linalg.norm(ring - center, axis=1)))
        c = (int(round(center[0])), int(round(center[1])))
        cv2.circle(img, c, max(1, int(round(radius))), IRIS_COLOR, 1, cv2.LINE_AA)
        cv2.circle(img, c, 2, IRIS_CENTER_COLOR, -1, cv2.LINE_AA)


def _draw_panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
    sub = img[y : y + h, x : x + w]
    if sub.size == 0:
        return
    panel = np.full_like(sub, PANEL)
    cv2.addWeighted(panel, alpha, sub, 1 - alpha, 0, dst=sub)


def draw_radar(img: np.ndarray, direction: str, h: Optional[float], v: Optional[float],
               origin: tuple[int, int], cell: int = 22, mirrored: bool = True,
               vertical_center: float = 0.0) -> None:
    """3x3 grid with the active cell highlighted plus a dot for the raw gaze."""
    x0, y0 = origin
    size = cell * 3
    _draw_panel(img, x0 - 6, y0 - 6, size + 12, size + 12)
    for i in range(4):
        cv2.line(img, (x0, y0 + i * cell), (x0 + size, y0 + i * cell), DIM_COLOR, 1)
        cv2.line(img, (x0 + i * cell, y0), (x0 + i * cell, y0 + size), DIM_COLOR, 1)

    cells = {g.CENTER: (1, 1), g.LEFT: (0, 1), g.RIGHT: (2, 1), g.UP: (1, 0), g.DOWN: (1, 2)}
    cx, cy = cells.get(direction, (1, 1))
    cv2.rectangle(
        img,
        (x0 + cx * cell + 2, y0 + cy * cell + 2),
        (x0 + (cx + 1) * cell - 2, y0 + (cy + 1) * cell - 2),
        ACCENT if direction != g.CENTER else EYE_COLOR,
        -1,
    )

    if h is not None and v is not None:
        user_h = h if mirrored else 1.0 - h
        # Map h in ~[0.3, 0.7] and v in ~centre +/- 0.12 onto the grid.
        px = (user_h - 0.5) / 0.2 * 0.5 + 0.5
        py = (v - vertical_center) / 0.12 * 0.5 + 0.5
        px = float(np.clip(px, 0.0, 1.0))
        py = float(np.clip(py, 0.0, 1.0))
        dot = (int(x0 + px * size), int(y0 + py * size))
        cv2.circle(img, dot, 4, IRIS_CENTER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, dot, 4, (255, 255, 255), 1, cv2.LINE_AA)


def velocity_label(velocity: tuple[float, float]) -> str:
    """ASCII arrow plus speed, e.g. ``<- 420 px/s``; ``stop`` when still."""
    vx, vy = velocity
    speed = float(np.hypot(vx, vy))
    if speed < 1.0:
        return "stop"
    horiz = "<" if vx < 0 else ">" if vx > 0 else ""
    vert = "^" if vy < 0 else "v" if vy > 0 else ""
    if abs(vx) < 0.5 * abs(vy):
        horiz = ""
    elif abs(vy) < 0.5 * abs(vx):
        vert = ""
    return f"{horiz}{vert} {speed:.0f} px/s"


def draw_mouse_line(img: np.ndarray, mouse: MouseHud, org: tuple[int, int]) -> None:
    if not mouse.enabled:
        text, color = "MOUSE OFF  m to resume", DIM_COLOR
    elif mouse.released:
        text, color = "MOUSE ON   no face", ALERT
    elif mouse.frozen:
        text, color = "MOUSE ON   hold", ACCENT
    else:
        text, color = f"MOUSE ON   {velocity_label(mouse.velocity)}", EYE_COLOR
    if mouse.dry_run:
        text += " (dry)"
    _put_text(img, text, org, 0.5, color)


def draw_click_flash(img: np.ndarray, button: str) -> None:
    w_img = img.shape[1]
    msg = f"CLICK {button.upper()}"
    (tw, th), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
    x = (w_img - tw) // 2
    _draw_panel(img, x - 12, 14, tw + 24, th + 24, alpha=0.7)
    _put_text(img, msg, (x, 14 + th + 12), 1.0, ACCENT, 2)


def draw_hud(img: np.ndarray, state: HudState) -> None:
    h_img, w_img = img.shape[:2]

    panel_w, panel_h = (250, 112) if state.mouse is None else (300, 136)
    _draw_panel(img, 8, 8, panel_w, panel_h)
    if state.face_found:
        color = ACCENT if state.direction != g.CENTER else EYE_COLOR
        _put_text(img, f"GAZE  {state.direction}", (16, 38), 0.8, color, 2)
        hv = (
            f"h={state.h:.2f}  v={state.v:+.2f}"
            if state.h is not None and state.v is not None
            else "h=--   v=--"
        )
        _put_text(img, hv, (16, 62), 0.5, DIM_COLOR)
    else:
        _put_text(img, "GAZE  --", (16, 38), 0.8, DIM_COLOR, 2)
        _put_text(img, "h=--   v=--", (16, 62), 0.5, DIM_COLOR)

    ear_txt = f"EAR {state.ear:.2f}" if state.ear is not None else "EAR --"
    closed = "  (closed)" if state.eyes_closed else ""
    _put_text(img, f"Blinks {state.blink_count}   {ear_txt}{closed}", (16, 86), 0.5, TEXT_COLOR)
    _put_text(img, f"{state.fps:4.1f} fps", (16, 108), 0.5, DIM_COLOR)
    if state.mouse is not None:
        draw_mouse_line(img, state.mouse, (16, 132))
        if state.mouse.click_flash:
            draw_click_flash(img, state.mouse.click_flash)

    draw_radar(img, state.direction if state.face_found else g.CENTER,
               state.h if state.face_found else None,
               state.v if state.face_found else None,
               origin=(w_img - 6 * 22 + 40, 20), mirrored=state.mirrored,
               vertical_center=state.vertical_center)

    if not state.face_found:
        banner_h = 36
        _draw_panel(img, 0, h_img - banner_h, w_img, banner_h, alpha=0.7)
        msg = "No face detected - move into the frame"
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
        _put_text(img, msg, ((w_img - tw) // 2, h_img - 11), 0.7, ALERT, 2)

    hint = "q quit | f mirror | l landmarks | r reset"
    if state.mouse is not None:
        hint += " | m mouse"
    (tw, _), _ = cv2.getTextSize(hint, FONT, 0.42, 1)
    _put_text(img, hint, (w_img - tw - 10, h_img - 10 if state.face_found else h_img - 46), 0.42, DIM_COLOR)


def render(img: np.ndarray, landmarks: Optional[np.ndarray], state: HudState) -> np.ndarray:
    if landmarks is not None and state.show_landmarks:
        draw_eyes(img, landmarks)
    draw_hud(img, state)
    return img
