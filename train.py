"""
Face gender classification on UTKFace.

UTKFace filenames encode the labels directly:
    [age]_[gender]_[race]_[timestamp].jpg.chip.jpg
      gender: 0 male, 1 female
      race:   0 White, 1 Black, 2 Asian, 3 Indian, 4 Others

We train on the aligned & cropped faces and, at the end, print accuracy broken
down by race x gender. A single headline number hides where the model actually
fails; the per-subgroup gap is the Gender Shades finding reproduced on a small
model, and it is the number worth showing.

The checkpoint format matches what realtime_fused.py / webcam.py load:
    {"state_dict", "model_name", "img_size"}.

Usage:
    pip install torch timm albumentations pillow numpy
    python train.py --data ./UTKFace --epochs 6        # -> gender_model.pt
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
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 160
GENDER_NAMES = {0: "male", 1: "female"}
RACE_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}


def parse_utkface(data_dir):
    """Return (train_records, val_records); each record is (path, gender, race).

    UTKFace has no official split, so we take a deterministic 90/10 shuffle.
    Files with malformed names (a handful in the release have missing fields)
    are skipped rather than crashing the run.
    """
    root = Path(data_dir)
    files = sorted(root.rglob("*.jpg"))
    if not files:
        raise SystemExit(f"No .jpg files found under {root} — point --data at the "
                         f"extracted UTKFace folder (aligned & cropped).")

    records, skipped = [], 0
    for path in files:
        parts = path.name.split("_")
        if len(parts) < 3:
            skipped += 1
            continue
        try:
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:
            skipped += 1
            continue
        if gender not in GENDER_NAMES or race not in RACE_NAMES:
            skipped += 1
            continue
        records.append((path, gender, race))

    if not records:
        raise SystemExit("Found .jpg files but none had parseable UTKFace names.")

    rng = random.Random(42)
    rng.shuffle(records)
    cut = int(len(records) * 0.9)
    train, val = records[:cut], records[cut:]
    print(f"images: {len(records)} usable ({skipped} skipped)  "
          f"train {len(train)}  val {len(val)}")
    return train, val


class FaceDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, gender, race = self.records[i]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, gender, race


def build_transforms():
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=12, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.CoarseDropout(max_holes=3, max_height=28, max_width=28, p=0.2),
        A.ImageCompression(quality_lower=60, p=0.2),
        norm, ToTensorV2(),
    ])
    eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), norm, ToTensorV2()])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    cells = defaultdict(lambda: [0, 0])   # key -> [hits, n]

    for imgs, genders, races in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs).argmax(1).cpu()
        hits = preds.eq(genders)
        correct += hits.sum().item()
        total += len(hits)
        for hit, g, r in zip(hits.tolist(), genders.tolist(), races.tolist()):
            for key in ((r, g), (r, None), (None, g)):
                cell = cells[key]
                cell[0] += int(hit)
                cell[1] += 1

    return correct / max(total, 1), cells


def print_subgroup_table(cells):
    print("\n" + "=" * 54)
    print("ACCURACY BY ETHNICITY x GENDER")
    print("=" * 54)
    print(f"{'group':<28}{'acc':>8}{'n':>10}")
    print("-" * 54)

    subgroup_accs = []
    for race in sorted(RACE_NAMES):
        agg = cells.get((race, None))
        if not agg or agg[1] == 0:
            continue
        print(f"{RACE_NAMES[race]:<28}{agg[0] / agg[1]:>7.1%}{agg[1]:>10}")
        for gender in (0, 1):
            cell = cells.get((race, gender))
            if cell and cell[1]:
                acc = cell[0] / cell[1]
                subgroup_accs.append((acc, f"{RACE_NAMES[race]} {GENDER_NAMES[gender]}"))
                label = f"  └ {GENDER_NAMES[gender]}"
                print(f"{label:<28}{acc:>7.1%}{cell[1]:>10}")

    print("-" * 54)
    for gender in (0, 1):
        agg = cells.get((None, gender))
        if agg and agg[1]:
            print(f"{'all ' + GENDER_NAMES[gender]:<28}{agg[0] / agg[1]:>7.1%}{agg[1]:>10}")

    if len(subgroup_accs) >= 2:
        best, worst = max(subgroup_accs), min(subgroup_accs)
        print("-" * 54)
        print(f"best  subgroup: {best[1]:<20}{best[0]:.1%}")
        print(f"worst subgroup: {worst[1]:<20}{worst[0]:.1%}")
        print(f"gap: {best[0] - worst[0]:.1%}  "
              f"-- this spread, not the headline number, is the honest result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to extracted UTKFace folder")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model", default="mobilenetv3_small_100")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="gender_model.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    train_recs, val_recs = parse_utkface(args.data)
    train_tf, eval_tf = build_transforms()
    train_loader = DataLoader(
        FaceDataset(train_recs, train_tf), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=(device == "cuda"),
        drop_last=True)
    val_loader = DataLoader(
        FaceDataset(val_recs, eval_tf), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=(device == "cuda"))

    # UTKFace is close to gender-balanced, but weight the loss anyway so a skew
    # in whatever subset you downloaded can't be gamed by predicting the prior.
    counts = np.bincount([g for _, g, _ in train_recs], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    print(f"class counts {counts.tolist()} -> loss weights {weights.tolist()}")

    model = timm.create_model(args.model, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (imgs, genders, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=use_amp):
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
    print_subgroup_table(cells)

    torch.save({"state_dict": model.state_dict(), "model_name": args.model,
                "img_size": IMG_SIZE}, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
