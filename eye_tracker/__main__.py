"""CLI entry point: `python -m eye_tracker` or `eye-tracker`."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2

from . import __version__
from .blink import BlinkDetector, both_eyes_ear
from .camera import CameraError, FrameSource
from .gaze import GazeClassifier, GazeThresholds, measure_gaze
from .landmarks import FaceLandmarker
from .logger import MovementLogger
from .mouse import DryRunBackend, GazeMouseController, MouseConfig, PynputBackend, detect_screen_size
from .overlay import HudState, MouseHud, render

WINDOW_TITLE = "Eye Tracker"
CLICK_FLASH_S = 0.6


def parse_screen(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        size = (int(w), int(h))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1920x1080") from None
    if size[0] <= 0 or size[1] <= 0:
        raise argparse.ArgumentTypeError("screen size must be positive")
    return size


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eye-tracker",
        description="Track eye movement through the webcam: iris overlay, gaze direction, blinks.",
    )
    src = p.add_argument_group("source")
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    src.add_argument("--video", type=str, default=None, help="run on a video file instead of the webcam")
    src.add_argument("--width", type=int, default=None, help="requested capture width")
    src.add_argument("--height", type=int, default=None, help="requested capture height")
    src.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    src.add_argument("--fast", action="store_true",
                     help="with --video: process as fast as possible instead of pacing to the file's frame rate")

    view = p.add_argument_group("display")
    view.add_argument("--no-flip", dest="flip", action="store_false", help="do not mirror the preview")
    view.add_argument("--no-window", action="store_true", help="headless: no preview window")
    view.add_argument("--no-landmarks", dest="landmarks", action="store_false", help="start with the overlay hidden")
    view.add_argument("--save-frame", type=str, default=None, help="write the last rendered frame to this image path")

    out = p.add_argument_group("logging")
    out.add_argument("--log-csv", type=str, default=None, help="append events to this CSV file")
    out.add_argument("--quiet", action="store_true", help="do not print the movement log")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--model", type=str, default=None, help="path to face_landmarker.task")
    tune.add_argument("--h-threshold", type=float, default=GazeThresholds.horizontal)
    tune.add_argument("--up-threshold", type=float, default=GazeThresholds.up)
    tune.add_argument("--down-threshold", type=float, default=GazeThresholds.down)
    tune.add_argument("--v-center", type=float, default=GazeThresholds.vertical_center)
    tune.add_argument("--hysteresis", type=float, default=GazeThresholds.hysteresis)
    tune.add_argument("--smoothing", type=float, default=0.35, help="EMA factor in (0, 1]; lower is smoother")
    tune.add_argument("--ear-threshold", type=float, default=0.21)
    tune.add_argument("--blink-frames", type=int, default=2, help="min closed frames to count a blink")
    tune.add_argument("--blink-max-frames", type=int, default=15,
                      help="closures longer than this are logged as 'eyes closed', not blinks")

    mouse = p.add_argument_group("mouse control (off by default)")
    mouse.add_argument("--mouse", action="store_true",
                       help="move the OS cursor with your gaze and click with deliberate blinks")
    mouse.add_argument("--dry-run", action="store_true",
                       help="run the mouse logic but only log the moves/clicks it would make (implies --mouse)")
    mouse.add_argument("--mouse-speed", type=float, default=MouseConfig.max_speed,
                       help=f"max cursor speed in px/s (default {MouseConfig.max_speed:.0f})")
    mouse.add_argument("--mouse-gain", type=float, default=MouseConfig.gain,
                       help=f"speed curve exponent; 1 = linear, higher = gentler near centre (default {MouseConfig.gain})")
    mouse.add_argument("--click-blink-min", type=float, default=MouseConfig.click_min_s,
                       help=f"shortest eye closure counted as a deliberate blink, seconds (default {MouseConfig.click_min_s})")
    mouse.add_argument("--click-blink-max", type=float, default=MouseConfig.click_max_s,
                       help=f"longest eye closure counted as a deliberate blink, seconds (default {MouseConfig.click_max_s})")
    mouse.add_argument("--screen", type=parse_screen, default=None, metavar="WxH",
                       help="screen size in pixels if it cannot be detected, e.g. 1920x1080")

    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


class FpsMeter:
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._last: Optional[float] = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else (self._alpha * inst + (1 - self._alpha) * self.fps)
        self._last = now
        return self.fps


def run(args: argparse.Namespace) -> int:
    thresholds = GazeThresholds(
        horizontal=args.h_threshold,
        up=args.up_threshold,
        down=args.down_threshold,
        vertical_center=args.v_center,
        hysteresis=args.hysteresis,
    )
    gaze = GazeClassifier(thresholds=thresholds, smoothing=args.smoothing, mirrored=args.flip)
    blink = BlinkDetector(threshold=args.ear_threshold, min_frames=args.blink_frames, max_frames=args.blink_max_frames)
    log = MovementLogger(csv_path=args.log_csv, quiet=args.quiet)
    fps = FpsMeter()
    show_landmarks = args.landmarks
    mirrored = args.flip
    show_window = not args.no_window
    last_frame = None

    mouse: Optional[GazeMouseController] = None
    if args.mouse or args.dry_run:
        if args.click_blink_min <= 0 or args.click_blink_max <= args.click_blink_min:
            print("error: need 0 < --click-blink-min < --click-blink-max", file=sys.stderr)
            return 2
        mouse_config = MouseConfig(
            max_speed=args.mouse_speed,
            gain=args.mouse_gain,
            dead_zone_h=thresholds.horizontal,
            dead_zone_up=thresholds.up,
            dead_zone_down=thresholds.down,
            vertical_center=thresholds.vertical_center,
            click_min_s=args.click_blink_min,
            click_max_s=args.click_blink_max,
        )
        try:
            if args.dry_run:
                screen = args.screen or detect_screen_size() or (1920, 1080)
                backend = DryRunBackend(screen=screen, emit=log.info)
            else:
                backend = PynputBackend(screen=args.screen)
        except Exception as exc:
            print(
                f"error: could not set up mouse control: {exc}\n"
                "Try --dry-run to test without touching the mouse, or --screen WxH if the screen size is unknown.",
                file=sys.stderr,
            )
            return 2
        mouse = GazeMouseController(backend, config=mouse_config, mirrored=args.flip)
        sw, sh = backend.screen_size()
        log.info(
            f"mouse control {'DRY RUN' if args.dry_run else 'ON'}: screen {sw}x{sh}, "
            f"max {mouse_config.max_speed:.0f} px/s, click blink {mouse_config.click_min_s:.2f}-"
            f"{mouse_config.click_max_s:.2f} s, press m to pause"
        )

    try:
        landmarker = FaceLandmarker(model_path=args.model)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        with FrameSource(camera=args.camera, video=args.video, width=args.width, height=args.height) as source:
            w, h = source.size
            log.info(f"source opened: {'file ' + args.video if args.video else f'webcam #{args.camera}'} {w}x{h}")
            if show_window:
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            # A file is decoded far faster than real time; pace it so the
            # preview looks like a live camera. Headless runs stay fast.
            pace = source.is_file and show_window and not args.fast
            playback_start = time.perf_counter()

            for frame in source.frames():
                if args.max_frames is not None and frame.index >= args.max_frames:
                    break
                if pace:
                    due = playback_start + frame.timestamp_ms / 1000.0
                    delay = due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                img = cv2.flip(frame.image, 1) if mirrored else frame.image
                landmarks = landmarker.detect(img, frame.timestamp_ms)
                fps.tick()
                now_s = frame.timestamp_ms / 1000.0

                if landmarks is None:
                    log.face(False)
                    blink.reset()
                    gs = None
                    state = HudState(
                        direction=gaze.direction, h=None, v=None, ear=None,
                        blink_count=blink.count, eyes_closed=False, fps=fps.fps,
                        face_found=False, show_landmarks=show_landmarks, mirrored=mirrored,
                        vertical_center=thresholds.vertical_center,
                    )
                else:
                    log.face(True)
                    right_ear, left_ear = both_eyes_ear(landmarks)
                    ear = (right_ear + left_ear) / 2.0
                    event = blink.update(ear)
                    if event is not None:
                        if event.is_blink:
                            log.blink(event.count, event.duration_frames, ear=ear)
                        else:
                            log.eyes_closed(event.duration_frames, ear=ear, blinks=blink.count)

                    measure = measure_gaze(landmarks)
                    gs = gaze.update(measure, eyes_closed=blink.is_closed) if measure is not None else None
                    if gs is not None and not blink.is_closed:
                        log.gaze(gs.direction, gs.h, gs.v, ear=ear, blinks=blink.count)

                    state = HudState(
                        direction=gs.direction if gs else gaze.direction,
                        h=gs.h if gs else None, v=gs.v if gs else None, ear=ear,
                        blink_count=blink.count, eyes_closed=blink.is_closed, fps=fps.fps,
                        face_found=True, show_landmarks=show_landmarks, mirrored=mirrored,
                        vertical_center=thresholds.vertical_center,
                    )

                if mouse is not None:
                    if gs is None:
                        upd = mouse.update(None, None, False, now_s)
                    else:
                        upd = mouse.update(gs.h, gs.v, blink.is_closed, now_s)
                    if upd.click is not None and upd.position is not None:
                        log.click(upd.click.button, upd.position[0], upd.position[1], blinks=blink.count)
                    if upd.released_changed:
                        log.info("mouse: face lost, control released" if upd.released else "mouse: face back, control resumed")
                    flash = mouse.last_click
                    state.mouse = MouseHud(
                        enabled=mouse.enabled, velocity=upd.velocity, frozen=upd.frozen,
                        released=upd.released, dry_run=args.dry_run,
                        click_flash=flash.button if flash and now_s - flash.t < CLICK_FLASH_S else None,
                    )

                if show_window or args.save_frame:
                    rendered = render(img, landmarks, state)
                    last_frame = rendered
                    if show_window:
                        cv2.imshow(WINDOW_TITLE, rendered)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key == ord("f"):
                            mirrored = not mirrored
                            gaze.mirrored = mirrored
                            if mouse is not None:
                                mouse.mirrored = mirrored
                            log.info(f"mirror view {'on' if mirrored else 'off'}")
                        elif key == ord("l"):
                            show_landmarks = not show_landmarks
                        elif key == ord("r"):
                            gaze.reset()
                            blink.count = 0
                            blink.reset()
                            if mouse is not None:
                                mouse.reset()
                            log.info("reset")
                        elif key == ord("m") and mouse is not None:
                            mouse.enabled = not mouse.enabled
                            log.info(f"mouse control {'resumed' if mouse.enabled else 'paused'}")
                        try:
                            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        except cv2.error:
                            break
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        log.info(f"stopped. blinks={blink.count}")
        log.close()
        if show_window:
            cv2.destroyAllWindows()
        if args.save_frame and last_frame is not None:
            cv2.imwrite(args.save_frame, last_frame)
            print(f"saved last frame to {args.save_frame}", file=sys.stderr)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
