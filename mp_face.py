"""Thin wrapper over MediaPipe's Tasks FaceDetector.

The legacy `mp.solutions.face_detection` API was removed in mediapipe 0.10.3x,
so we use the Tasks API instead. The wrapper keeps a tiny, stable interface —
`detect(frame_bgr)` returns a list of `((x1, y1, x2, y2), score)` in pixel
coords — so the rest of the pipeline doesn't care which API is underneath.

The model file (blaze_face_short_range.tflite, ~220 KB) is fetched once on first
use and cached next to this file. Short-range is the right pick for a webcam:
it targets faces within ~2 m of the camera.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
              "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "blaze_face_short_range.tflite")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print("downloading blaze_face_short_range.tflite (~220 KB) ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceDetector:
    """Drop-in face detector. Use as a context manager or call .close()."""

    def __init__(self, min_confidence=0.5):
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            min_detection_confidence=min_confidence)
        self._det = vision.FaceDetector.create_from_options(opts)

    def detect(self, frame_bgr):
        """Return [((x1, y1, x2, y2), score), ...] in pixel coordinates."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        out = []
        for d in self._det.detect(image).detections:
            b = d.bounding_box
            x1, y1 = int(b.origin_x), int(b.origin_y)
            x2, y2 = x1 + int(b.width), y1 + int(b.height)
            score = float(d.categories[0].score) if d.categories else 1.0
            out.append(((x1, y1, x2, y2), score))
        return out

    def close(self):
        self._det.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
