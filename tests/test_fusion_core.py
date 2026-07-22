import numpy as np

from fusion_core import Tracker, containment, fuse, iou, upper_body_box


def test_iou_handles_identity_overlap_and_separation():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert np.isclose(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


def test_containment_and_upper_body_geometry():
    assert containment((2, 2, 8, 8), (0, 0, 10, 10)) == 1.0
    assert containment((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5
    assert upper_body_box((10, 20, 50, 100)) == (10, 20, 50, 60)


def test_fuse_normalizes_reliability_weights():
    face = np.array([0.8, 0.2])
    body = np.array([0.2, 0.8])

    probs, weights = fuse(face, 1.0, 48, body, 1.0)

    assert np.allclose(weights, [0.625, 0.375])
    assert np.allclose(probs, [0.575, 0.425])


def test_fuse_abstains_when_no_signal_has_weight():
    probs, weights = fuse(None, 0.0, 0, None, 0.0)
    assert probs is None
    assert weights is None


def test_tracker_keeps_identity_and_smooths_probabilities():
    tracker = Tracker()
    assert tracker.update([(0, 0, 100, 100)]) == [(0, 0)]
    assert tracker.update([(5, 0, 105, 100)]) == [(0, 0)]

    first = tracker.smooth(0, np.array([0.8, 0.2]))
    second = tracker.smooth(0, np.array([0.4, 0.6]))

    assert np.allclose(first, [0.8, 0.2])
    assert np.allclose(second, [0.6, 0.4])
    assert np.isclose(tracker.ease(0, 0.9), 0.6)
