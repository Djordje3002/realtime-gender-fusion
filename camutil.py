"""Robust webcam opening for macOS (and elsewhere).

Two things bite on a Mac:
  * OpenCV's default backend selection can skip AVFoundation and fail, so we
    ask for it explicitly first, then fall back to whatever OpenCV picks.
  * Continuity Camera means an iPhone shows up as a second device, so the
    built-in camera isn't always index 0. If the requested index won't open we
    probe the next few.

`open_camera(index)` returns an opened cv2.VideoCapture, or None.
"""

import cv2

# AVFoundation is the native macOS backend; CAP_ANY lets OpenCV choose.
_BACKENDS = [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]


def _try(index):
    for backend in _BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()          # confirm we can actually pull a frame
            if ok:
                return cap
        cap.release()
    return None


def open_camera(index=0, probe=4):
    """Open `index`; if that fails, probe indices 0..probe-1 as a fallback."""
    cap = _try(index)
    if cap is not None:
        return cap
    for alt in range(probe):
        if alt == index:
            continue
        cap = _try(alt)
        if cap is not None:
            print(f"camera {index} unavailable; using camera {alt} instead")
            return cap
    return None
