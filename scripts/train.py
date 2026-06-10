"""Train a Pokemon classifier (Gen 1+2) on the scraped dataset.

Modernized successor to train_baseline_model.py:
  - ConvNeXt-Tiny backbone (ImageNet pre-trained), re-headed for N classes
  - argparse config instead of hardcoded paths
  - stratified train/val/test split (guarantees per-class coverage)
  - strong augmentation (RandAugment, color jitter, random erasing)
  - mixed-precision (AMP) + cosine LR schedule on GPU
  - early stopping on val accuracy
  - TensorBoard logging
  - confusion matrix + per-class accuracy report on the held-out test set

Two phases (the same script, different flags):
  # Phase 1 - color baseline:
  python train.py --data-dir data/raw --epochs 30 --out best_color.pth

  # Phase 2 - randomized sketch fine-tune (warm-start from phase 1):
  python train.py --data-dir data/raw --epochs 15 --sketch \
      --init-weights best_color.pth --lr 5e-5 --out best_sketch.pth

NOTE: trains with ImageNet normalization (mean/std below). The inference app
(classifier.py) must use the SAME normalization for the new checkpoint.
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

from sketch import RandomizedSketch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_transforms(img_size, sketch):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_tf = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip()]
    eval_tf = [transforms.Resize(img_size + 32),
               transforms.CenterCrop(img_size)]

    if sketch:
        # Drawn-style augmentation; applied before tensor conversion.
        train_tf.append(RandomizedSketch(p=0.9))
        eval_tf.append(RandomizedSketch(p=1.0, styles=("pencil",)))
    else:
        train_tf += [transforms.RandAugment(),
                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]

    train_tf += [transforms.ToTensor(), norm, transforms.RandomErasing(p=0.25)]
    eval_tf += [transforms.ToTensor(), norm]
    return transforms.Compose(train_tf), transforms.Compose(eval_tf)


def stratified_split(targets, val_frac, test_frac, seed):
    """Return train/val/test index lists with per-class proportions preserved."""
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for t, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n > 3 else 0
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test:n_test + n_val]
        train_idx += idxs[n_test + n_val:]
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(backbone, num_classes, freeze_backbone=False):
    if backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
        head_params = model.classifier.parameters()
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
        head_params = model.classifier.parameters()
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head_params = model.fc.parameters()
    else:
        raise ValueError(f"unknown backbone {backbone}")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return model


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate_test(model, loader, device, classes, out_dir):
    """Per-class accuracy + confusion matrix on the test set."""
    model.eval()
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for inputs, labels in loader:
        inputs = inputs.to(device)
        preds = model(inputs).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            cm[t, p] += 1

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    acc = per_class_correct.sum() / max(1, per_class_total.sum())

    # Save confusion matrix
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + classes)
        for i, row in enumerate(cm):
            w.writerow([classes[i]] + row.tolist())

    # Per-class accuracy report, worst first
    report_path = os.path.join(out_dir, "per_class_accuracy.csv")
    rows = []
    for i, c in enumerate(classes):
        tot = per_class_total[i]
        a = per_class_correct[i] / tot if tot else 0.0
        rows.append((c, tot, a))
    rows.sort(key=lambda r: r[2])
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "test_samples", "accuracy"])
        for c, tot, a in rows:
            w.writerow([c, int(tot), f"{a:.4f}"])

    print(f"\nTest top-1 accuracy: {acc * 100:.2f}%")
    print("Weakest 10 classes:")
    for c, tot, a in rows[:10]:
        print(f"  {c:<14} {a * 100:5.1f}%  (n={int(tot)})")
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved per-class report -> {report_path}")
    return acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train PokeGuess Gen 1+2 classifier")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--backbone", default="convnext_tiny",
                    choices=["convnext_tiny", "efficientnet_b0", "resnet50"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6, help="Early-stop patience (epochs).")
    ap.add_argument("--sketch", action="store_true", help="Phase 2: randomized sketch aug.")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--init-weights", default=None, help="Warm-start checkpoint (.pth).")
    ap.add_argument("--out", default="pokeguess_gen2.pth")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  backbone: {args.backbone}  |  sketch: {args.sketch}")

    # Two ImageFolders over the same dir: one with train aug, one with eval aug.
    train_tf, eval_tf = build_transforms(args.img_size, args.sketch)
    train_full = datasets.ImageFolder(args.data_dir, transform=train_tf)
    eval_full = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    classes = train_full.classes
    print(f"Found {len(classes)} classes, {len(train_full)} images total")

    # Persist class list alongside the checkpoint (inference needs this order).
    class_list_path = os.path.splitext(args.out)[0] + "_classes.txt"
    with open(class_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")

    tr_idx, va_idx, te_idx = stratified_split(
        train_full.targets, args.val_frac, args.test_frac, args.seed)
    train_ds = Subset(train_full, tr_idx)
    val_ds = Subset(eval_full, va_idx)
    test_ds = Subset(eval_full, te_idx)
    print(f"Split -> train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # Weighted sampling to counter class imbalance (~12x here): each sample is
    # drawn with probability inversely proportional to its class frequency, so
    # every Pokemon is seen roughly equally per epoch without discarding data.
    train_targets = [train_full.targets[i] for i in tr_idx]
    class_counts = np.bincount(train_targets, minlength=len(classes))
    class_weight = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weight[t] for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    model = build_model(args.backbone, len(classes), args.freeze_backbone).to(device)
    if args.init_weights:
        state = torch.load(args.init_weights, map_location=device)
        # Tolerate a different classifier head (e.g. warm-starting a 386-class
        # model from a 251-class checkpoint): load every tensor whose shape
        # matches, leave the rest (the new head) at their fresh init.
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        skipped = len(model_sd) - len(filtered)
        print(f"Warm-started from {args.init_weights} "
              f"({len(filtered)} tensors loaded, {skipped} left at fresh init)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    writer = SummaryWriter(args.out_dir)

    best_acc, epochs_no_improve = 0.0, 0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer=optimizer, scaler=scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch+1:>2}/{args.epochs}] "
              f"train loss {tr_loss:.3f} acc {tr_acc:5.2f}% | "
              f"val loss {va_loss:.3f} acc {va_acc:5.2f}% | lr {lr_now:.2e}")
        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        if va_acc > best_acc:
            best_acc = va_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.out)
            print(f"  saved best ({best_acc:.2f}%) -> {args.out}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping (no val improvement in {args.patience} epochs).")
                break

    # Final test-set evaluation with the best checkpoint.
    model.load_state_dict(torch.load(args.out, map_location=device))
    evaluate_test(model, test_loader, device, classes, args.out_dir)
    writer.close()
    print(f"\nDone. Best val acc {best_acc:.2f}%. Checkpoint: {args.out}")
    print(f"Class list: {class_list_path}")


if __name__ == "__main__":
    main()
