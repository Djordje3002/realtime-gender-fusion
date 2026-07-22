"""
Realtime fused gender classification.

    frame ─┬─ YOLO person detection ─→ upper-body crop ─→ body model ─┐
           └─ MediaPipe face detection ─→ face crop ────→ face model ─┴─→ fuse
                                                                        │
                                    IoU tracker + rolling average ──────┘

The fusion is confidence-weighted, not a flat 50/50: a large frontal face
dominates, and the body model carries the prediction when the person is turned
away or too far for the face detector. Per-track smoothing removes the
frame-to-frame flicker you get from independent per-frame inference.

Usage:
    pip install ultralytics opencv-python mediapipe torch timm numpy
    python realtime_fused.py --face-weights gender_model.pt --body-weights body_model.pt
    (q to quit, f to toggle the per-model debug readout)

Either model may be omitted; the pipeline degrades to whichever is present.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

from camutil import open_camera
from fusion_core import Tracker, containment, fuse, upper_body_box
from mp_face import FaceDetector

LABELS = ["male", "female"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MALE_COLOR = (230, 150, 60)     # BGR — left end of the scale
FEMALE_COLOR = (80, 120, 250)   # BGR — right end of the scale
NEUTRAL = (235, 235, 235)


# ─────────────────────────── models ───────────────────────────

def load_model(weights, device):
    if not weights:
        return None, None
    if not Path(weights).is_file():
        print(f"weights not found: {weights} — disabling that classifier")
        return None, None
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt.get("img_size", 160)


def preprocess(crop, size):
    img = cv2.resize(crop, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1))


@torch.no_grad()
def classify_batch(model, crops, size, device):
    """One forward pass for all crops of a given type. Returns (N, 2) probs."""
    if model is None or not crops:
        return None
    batch = torch.stack([preprocess(c, size) for c in crops]).to(device)
    return torch.softmax(model(batch), dim=1).cpu().numpy()


# ─────────────────────────── geometry ───────────────────────────

def crop(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


# ─────────────────────────── scale overlay ───────────────────────────

_GRADIENT_CACHE = {}


def gradient_strip(width, height):
    """Male-to-female colour ramp, cached per size — rebuilding this every
    frame for every person is the easy way to tank your framerate."""
    key = (width, height)
    if key not in _GRADIENT_CACHE:
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        left = np.array(MALE_COLOR, dtype=np.float32)
        right = np.array(FEMALE_COLOR, dtype=np.float32)
        strip = (left * (1.0 - ramp) + right * ramp).astype(np.uint8)
        _GRADIENT_CACHE[key] = np.repeat(strip, height, axis=0)
    return _GRADIENT_CACHE[key]


def draw_panel(frame, x1, y1, x2, y2, alpha=0.5):
    """Darken a rectangle of the frame so an overlaid gauge/text stays readable
    against a bright or busy background."""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1.0 - alpha, 0, roi)


def draw_label(frame, text, org, scale, color, thick=2, align="left"):
    """Text with a black outline so it reads on light or dark video.
    align anchors the given x: 'left' | 'center' | 'right'."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = int(org[0]), int(org[1])
    if align == "center":
        x -= tw // 2
    elif align == "right":
        x -= tw
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return tw, th


def draw_scale(frame, cx, top, width, height, p_female, confident, dead_zone=0.0):
    """Horizontal slider: marker sits at p_female, so left = male, right =
    female, dead centre = the model has nothing to say."""
    fh, fw = frame.shape[:2]
    x0 = int(max(2, min(fw - width - 2, cx - width // 2)))
    y0 = int(max(2, min(fh - height - 2, top)))

    roi = frame[y0:y0 + height, x0:x0 + width]
    if roi.shape[:2] != (height, width):
        return

    alpha = 0.95 if confident else 0.55
    cv2.addWeighted(gradient_strip(width, height), alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + width + 1, y0 + height + 1), (20, 20, 20), 2)

    # shade the abstain band so "uncertain" is visible, not just implied
    if dead_zone > 0:
        half = int(width * dead_zone / 2)
        band = frame[y0:y0 + height, x0 + width // 2 - half: x0 + width // 2 + half]
        if band.size:
            band[:] = (band * 0.45).astype(np.uint8)

    cv2.line(frame, (x0 + width // 2, y0 - 3),
             (x0 + width // 2, y0 + height + 3), (200, 200, 200), 1)

    mx = int(np.clip(x0 + p_female * width, x0, x0 + width - 1))
    # dark halo first, bright core on top -> the marker reads on any colour
    cv2.line(frame, (mx, y0 - 6), (mx, y0 + height + 6), (0, 0, 0), 5)
    cv2.line(frame, (mx, y0 - 6), (mx, y0 + height + 6), NEUTRAL, 2)
    tri = np.array([[mx, y0 - 5], [mx - 7, y0 - 16], [mx + 7, y0 - 16]], np.int32)
    cv2.fillPoly(frame, [tri], NEUTRAL)
    cv2.polylines(frame, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)


# ─────────────────────────── main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-weights", default="gender_model.pt")
    ap.add_argument("--body-weights", default="body_model.pt")
    ap.add_argument("--detector", default="yolov8n.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--scale", default="both",
                    choices=["both", "person", "primary", "off"],
                    help="both = per-person sliders + bottom gauge; "
                         "person = per-person only; primary = bottom gauge only")
    args = ap.parse_args()

    from ultralytics import YOLO

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    face_model, face_size = load_model(args.face_weights, device)
    body_model, body_size = load_model(args.body_weights, device)
    if face_model is None and body_model is None:
        raise SystemExit("Need at least one of --face-weights / --body-weights")
    print(f"device={device}  face={face_model is not None}  body={body_model is not None}")

    detector = YOLO(args.detector)
    face_detector = FaceDetector(min_confidence=0.5)
    tracker = Tracker()
    show_debug = False

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS: grant your terminal "
            f"camera access in System Settings > Privacy & Security > Camera, quit "
            f"the terminal fully (Cmd+Q), reopen, and retry.")

    with face_detector as faces:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 1. people
            result = detector(frame, classes=[0], conf=args.person_conf, verbose=False)[0]
            person_boxes, person_confs = [], []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_boxes.append((x1, y1, x2, y2))
                person_confs.append(float(box.conf[0]))

            # 2. faces (once for the whole frame, then matched to people)
            face_boxes, face_confs = [], []
            for fbox, fscore in faces.detect(frame):
                face_boxes.append(fbox)
                face_confs.append(fscore)

            face_for_person = {}
            for f_idx, fbox in enumerate(face_boxes):
                best_p, best_score = None, 0.5
                for p_idx, pbox in enumerate(person_boxes):
                    score = containment(fbox, pbox)
                    if score > best_score:
                        best_p, best_score = p_idx, score
                if best_p is not None and best_p not in face_for_person:
                    face_for_person[best_p] = f_idx

            # 3. batch both models
            body_crops, body_owner = [], []
            for p_idx, pbox in enumerate(person_boxes):
                c = crop(frame, upper_body_box(pbox))
                if c is not None:
                    body_crops.append(c)
                    body_owner.append(p_idx)

            face_crops, face_owner = [], []
            for p_idx, f_idx in face_for_person.items():
                c = crop(frame, face_boxes[f_idx])
                if c is not None:
                    face_crops.append(c)
                    face_owner.append(p_idx)

            body_probs = classify_batch(body_model, body_crops, body_size, device)
            face_probs = classify_batch(face_model, face_crops, face_size, device)

            by_person = {i: {} for i in range(len(person_boxes))}
            if body_probs is not None:
                for k, p_idx in enumerate(body_owner):
                    by_person[p_idx]["body"] = body_probs[k]
            if face_probs is not None:
                for k, p_idx in enumerate(face_owner):
                    by_person[p_idx]["face"] = face_probs[k]

            # 4. track, fuse, smooth, draw
            primary = (0, 0.5, False, "", NEUTRAL)  # area, p_female, conf, call, colour
            for tid, p_idx in tracker.update(person_boxes):
                pbox = person_boxes[p_idx]
                got = by_person[p_idx]

                f_idx = face_for_person.get(p_idx)
                face_conf = face_confs[f_idx] if f_idx is not None else 0.0
                face_px = (face_boxes[f_idx][2] - face_boxes[f_idx][0]) if f_idx is not None else 0

                fused, weights = fuse(got.get("face"), face_conf, face_px,
                                      got.get("body"), person_confs[p_idx])
                if fused is None:
                    continue

                smoothed = tracker.smooth(tid, fused)
                idx = int(np.argmax(smoothed))
                conf = float(smoothed[idx])

                if conf < args.min_conf:
                    text, color = f"#{tid} uncertain", (0, 165, 255)
                else:
                    text, color = f"#{tid} {LABELS[idx]} {conf:.0%}", (0, 200, 0)

                p_female = tracker.ease(tid, float(smoothed[1]))
                dead_zone = 2 * args.min_conf - 1.0   # width of the abstain band

                cv2.rectangle(frame, pbox[:2], pbox[2:], color, 2)
                cv2.putText(frame, text, (pbox[0], max(20, pbox[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if args.scale in ("both", "person"):
                    bar_w = int(np.clip(pbox[2] - pbox[0], 90, 240))
                    draw_scale(frame, (pbox[0] + pbox[2]) // 2, pbox[3] + 14,
                               bar_w, 10, p_female, conf >= args.min_conf, dead_zone)

                area = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                if area > primary[0]:
                    if conf < args.min_conf:
                        call, call_color = "UNCERTAIN", (0, 165, 255)
                    else:
                        call = f"{LABELS[idx].upper()}  {conf:.0%}"
                        call_color = MALE_COLOR if idx == 0 else FEMALE_COLOR
                    primary = (area, p_female, conf >= args.min_conf, call, call_color)

                if f_idx is not None:
                    fb = face_boxes[f_idx]
                    cv2.rectangle(frame, fb[:2], fb[2:], (255, 180, 0), 1)

                if show_debug:
                    lines = []
                    if "face" in got:
                        lines.append(f"face {LABELS[int(np.argmax(got['face']))]} "
                                     f"{got['face'].max():.0%}")
                    if "body" in got:
                        lines.append(f"body {LABELS[int(np.argmax(got['body']))]} "
                                     f"{got['body'].max():.0%}")
                    if weights:
                        lines.append("w " + "/".join(f"{x:.2f}" for x in weights))
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (pbox[0], pbox[1] + 18 + i * 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # 5. big gauge for the nearest person
            if primary[0] > 0 and args.scale in ("both", "primary"):
                _, p_female_big, confident_big, call, call_color = primary
                gh = 30
                gw = int(min(560, w - 220))            # leave room for side labels
                gy = h - 74                             # gauge top
                cx = w // 2

                # backdrop so the whole readout stays legible on any scene
                draw_panel(frame, cx - gw // 2 - 110, gy - 46,
                           cx + gw // 2 + 110, gy + gh + 16, alpha=0.5)

                # current call, centred above the gauge
                draw_label(frame, call, (cx, gy - 20), 0.95, call_color, 2, "center")

                draw_scale(frame, cx, gy, gw, gh, p_female_big, confident_big,
                           2 * args.min_conf - 1.0)

                # MALE / FEMALE end labels, bold + outlined, centred on the bar
                ty = gy + gh // 2 + 8
                draw_label(frame, "MALE", (cx - gw // 2 - 14, ty), 0.7, MALE_COLOR, 2, "right")
                draw_label(frame, "FEMALE", (cx + gw // 2 + 14, ty), 0.7, FEMALE_COLOR, 2, "left")

            cv2.imshow("fused classifier (q quit, f debug)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
