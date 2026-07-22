"""
Face-only gender classification — the simple version, no YOLO.

Use this to check your camera works before debugging the full fused pipeline,
and as the minimal demo when you have a face model but haven't trained the body
one. With no weights it just draws the face boxes, which is enough to confirm
MediaPipe and your webcam are alive.

Usage:
    python webcam.py                          # loads gender_model.pt if present
    python webcam.py --weights ""             # detection only, no classification
    python webcam.py --weights gender_model.pt --min-conf 0.7
    (q to quit)
"""

import argparse

import cv2
import numpy as np
import timm
import torch

from camutil import open_camera
from mp_face import FaceDetector

LABELS = ["male", "female"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights, device):
    if not weights:
        return None, None
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt.get("img_size", 160)


@torch.no_grad()
def classify(model, crop, size, device):
    img = cv2.resize(crop, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="gender_model.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.6)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model, size = load_model(args.weights, device)
    print(f"device={device}  classifier={'on' if model else 'off (detection only)'}")

    face_detector = FaceDetector(min_confidence=0.5)

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

            for (x1, y1, x2, y2), _score in faces.detect(frame):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue

                if model is None:
                    text, color = "face", (255, 180, 0)
                else:
                    idx, conf = classify(model, frame[y1:y2, x1:x2], size, device)
                    if conf < args.min_conf:
                        text, color = "uncertain", (0, 165, 255)
                    else:
                        text, color = f"{LABELS[idx]} {conf:.0%}", (0, 200, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("face gender (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
