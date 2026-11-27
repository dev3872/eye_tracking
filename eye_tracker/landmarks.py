"""MediaPipe Face Landmarker wrapper with automatic model download."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"
NUM_LANDMARKS = 478  # 468 face mesh points + 10 iris points

# Landmark indices (MediaPipe Face Mesh topology). "RIGHT"/"LEFT" are the
# subject's own right/left eye. In an un-mirrored camera image the subject's
# right eye appears on the image's left side.
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Six-point sets for Eye Aspect Ratio: [outer corner, upper1, upper2, inner corner, lower2, lower1]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [263, 387, 385, 362, 380, 373]

RIGHT_EYE_CORNERS = (33, 133)   # outer, inner
LEFT_EYE_CORNERS = (263, 362)   # outer, inner
RIGHT_EYE_LIDS = (159, 145)     # upper, lower
LEFT_EYE_LIDS = (386, 374)      # upper, lower

RIGHT_IRIS = [468, 469, 470, 471, 472]  # 468 = center
LEFT_IRIS = [473, 474, 475, 476, 477]   # 473 = center
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def ensure_model(model_path: Optional[str | os.PathLike] = None) -> Path:
    """Return the path to face_landmarker.task, downloading it on first run."""
    path = Path(model_path) if model_path else default_model_dir() / MODEL_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path
    if model_path:
        # An explicit path that does not exist is an error, not a download target.
        raise FileNotFoundError(f"Model file not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.part")
    print(f"Downloading Face Landmarker model to {path} ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", file=sys.stderr)
        print(file=sys.stderr)
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download the Face Landmarker model from {MODEL_URL}: {exc}\n"
            f"Download it manually and place it at {path}, or pass --model <path>."
        ) from exc
    return path


class FaceLandmarker:
    """Runs MediaPipe FaceLandmarker in VIDEO mode and returns pixel landmarks.

    `detect(frame_bgr, timestamp_ms)` returns an (478, 2) float32 array of
    pixel coordinates for the most prominent face, or None if no face is found.
    """

    def __init__(self, model_path: Optional[str | os.PathLike] = None, num_faces: int = 1) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        path = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        face = result.face_landmarks[0]
        pts = np.fromiter(
            (c for lm in face for c in (lm.x * w, lm.y * h)),
            dtype=np.float32,
            count=len(face) * 2,
        ).reshape(-1, 2)
        if pts.shape[0] < NUM_LANDMARKS:
            return None
        return pts

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
