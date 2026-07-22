"""
Upper-body gender classification on PA-100K.

PA-100K ships a single annotation.mat with 26 binary attributes per image.
We use attribute 0 ('Female') as the label, and attributes 4/5/6
(Front / Side / Back) as an evaluation grouping — because viewpoint is where
body-based classification actually falls apart, and that's the finding worth
showing.

IMPORTANT: we train on the TOP 50% of each pedestrian box, because that's what
the realtime pipeline feeds in. Train on full bodies and infer on upper bodies
and you get a domain mismatch that looks like a bug but isn't.

Usage:
    pip install torch timm albumentations scipy pillow numpy
    python train_body.py --data ./PA-100K --epochs 8

Expected layout:
    PA-100K/
      annotation/annotation.mat
      release_data/release_data/*.jpg
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 160
UPPER_FRACTION = 0.5          # keep top half of the pedestrian box
GENDER_NAMES = {0: "male", 1: "female"}
VIEW_NAMES = {0: "front", 1: "side", 2: "back", 3: "unknown"}
FEMALE_IDX = 0
VIEW_IDX = (4, 5, 6)          # Front, Side, Back


def load_pa100k(data_dir):
    """Return (train_records, val_records); each record is (path, gender, view)."""
    root = Path(data_dir)

    mat_candidates = [root / "annotation" / "annotation.mat", root / "annotation.mat"]
    mat_path = next((p for p in mat_candidates if p.exists()), None)
    if mat_path is None:
        raise SystemExit(f"annotation.mat not found under {root}")
    mat = loadmat(mat_path)

    # layout differs between the original release and the Kaggle mirror
    dir_candidates = [root / "release_data" / "release_data",
                      root / "release_data", root / "data", root]
    img_root = next((p for p in dir_candidates
                     if p.is_dir() and any(p.glob("*.jpg"))), None)
    if img_root is None:
        raise SystemExit(f"No image folder with .jpg files found under {root}")
    print(f"annotations: {mat_path}\nimages: {img_root}")

    def build(name_key, label_key):
        names = [str(n[0][0]) for n in mat[name_key]]
        labels = mat[label_key]
        out = []
        for name, attrs in zip(names, labels):
            path = img_root / name
            if not path.exists():
                continue
            gender = int(attrs[FEMALE_IDX])
            flags = [int(attrs[i]) for i in VIEW_IDX]
            view = flags.index(1) if sum(flags) == 1 else 3
            out.append((path, gender, view))
        return out

    train = build("train_images_name", "train_label")
    val = build("val_images_name", "val_label")
    print(f"train: {len(train)}   val: {len(val)}")
    if not train:
        raise SystemExit("No images resolved — check --data points at PA-100K root.")
    return train, val


class UpperBodyDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, gender, view = self.records[i]
        img = np.array(Image.open(path).convert("RGB"))
        cut = max(1, int(img.shape[0] * UPPER_FRACTION))
        img = self.transform(image=img[:cut])["image"]
        return img, gender, view


def build_transforms():
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.25),
        A.ImageCompression(quality_lower=55, p=0.25),
        norm, ToTensorV2(),
    ])
    eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), norm, ToTensorV2()])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    cells = defaultdict(lambda: [0, 0])

    for imgs, genders, views in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs).argmax(1).cpu()
        hits = preds.eq(genders)
        correct += hits.sum().item()
        total += len(hits)
        for hit, g, v in zip(hits.tolist(), genders.tolist(), views.tolist()):
            for key in ((v, g), (v, None)):
                cell = cells[key]
                cell[0] += int(hit)
                cell[1] += 1

    return correct / max(total, 1), cells


def print_viewpoint_table(cells):
    print("\n" + "=" * 54)
    print("ACCURACY BY VIEWPOINT")
    print("=" * 54)
    print(f"{'group':<28}{'acc':>8}{'n':>10}")
    print("-" * 54)

    for view in sorted(VIEW_NAMES):
        agg = cells.get((view, None))
        if not agg or agg[1] == 0:
            continue
        print(f"{VIEW_NAMES[view]:<28}{agg[0] / agg[1]:>7.1%}{agg[1]:>10}")
        for gender in (0, 1):
            cell = cells.get((view, gender))
            if cell and cell[1]:
                label = f"  └ {GENDER_NAMES[gender]}"
                print(f"{label:<28}{cell[0] / cell[1]:>7.1%}{cell[1]:>10}")

    front = cells.get((0, None))
    back = cells.get((2, None))
    if front and back and front[1] and back[1]:
        gap = front[0] / front[1] - back[0] / back[1]
        print("-" * 54)
        print(f"front - back gap: {gap:.1%}")
        print("The model leans on face/front cues far more than it admits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to PA-100K root")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model", default="mobilenetv3_small_100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="body_model.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_recs, val_recs = load_pa100k(args.data)
    train_tf, eval_tf = build_transforms()
    train_loader = DataLoader(
        UpperBodyDataset(train_recs, train_tf), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        UpperBodyDataset(val_recs, eval_tf), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)

    # PA-100K is male-skewed; weight the loss so the model can't coast on priors.
    counts = np.bincount([g for _, g, _ in train_recs], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    print(f"class counts {counts.tolist()} -> loss weights {weights.tolist()}")

    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (imgs, genders, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = criterion(model(imgs), genders)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}")

        acc, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val acc {acc:.2%}")

    acc, cells = evaluate(model, val_loader, device)
    print(f"\nFinal overall accuracy: {acc:.2%}")
    print_viewpoint_table(cells)

    torch.save({"state_dict": model.state_dict(), "model_name": args.model,
                "img_size": IMG_SIZE, "upper_fraction": UPPER_FRACTION}, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
