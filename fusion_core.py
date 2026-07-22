"""Pure geometry, fusion, and tracking primitives for the realtime demo.

This module intentionally depends only on NumPy and the standard library so
its behavior can be tested without camera, detector, or model dependencies.
"""

from collections import deque

import numpy as np


SMOOTH_WINDOW = 10
TRACK_MAX_AGE = 15
BODY_DISCOUNT = 0.6
FACE_REF_PX = 48.0
NEEDLE_EASE = 0.25


def iou(a, b):
    """Return intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def containment(inner, outer):
    """Return the fraction of ``inner`` that falls inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / max(area, 1)


def upper_body_box(person_box, fraction=0.5):
    """Return the upper fraction of a full-person bounding box."""
    x1, y1, x2, y2 = person_box
    return x1, y1, x2, y1 + int((y2 - y1) * fraction)


class Tracker:
    """Small greedy IoU tracker with rolling prediction smoothing."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes):
        for track in self.tracks.values():
            track["age"] += 1

        assigned, pairs = set(), []
        for det_idx, box in enumerate(boxes):
            best_id, best_iou = None, 0.3
            for tid, track in self.tracks.items():
                if tid in assigned:
                    continue
                score = iou(box, track["box"])
                if score > best_iou:
                    best_id, best_iou = tid, score
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "box": box,
                    "age": 0,
                    "needle": 0.5,
                    "history": deque(maxlen=SMOOTH_WINDOW),
                }
            assigned.add(best_id)
            self.tracks[best_id].update(box=box, age=0)
            pairs.append((best_id, det_idx))

        expired = [
            tid for tid, track in self.tracks.items()
            if track["age"] > TRACK_MAX_AGE
        ]
        for tid in expired:
            del self.tracks[tid]
        return pairs

    def smooth(self, tid, probs):
        history = self.tracks[tid]["history"]
        history.append(probs)
        return np.mean(history, axis=0)

    def ease(self, tid, target):
        track = self.tracks[tid]
        track["needle"] += (target - track["needle"]) * NEEDLE_EASE
        return track["needle"]


def fuse(face_probs, face_conf, face_px, body_probs, body_conf):
    """Return a confidence-weighted probability merge and normalized weights."""
    terms = []
    if face_probs is not None:
        size_factor = float(np.clip(face_px / FACE_REF_PX, 0.0, 1.0))
        terms.append((face_conf * size_factor, face_probs))
    if body_probs is not None:
        terms.append((body_conf * BODY_DISCOUNT, body_probs))

    terms = [(weight, probs) for weight, probs in terms if weight > 0.01]
    if not terms:
        return None, None

    total = sum(weight for weight, _ in terms)
    fused = sum(weight * probs for weight, probs in terms) / total
    return fused, [weight / total for weight, _ in terms]
