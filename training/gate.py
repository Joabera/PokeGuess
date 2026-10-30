"""Ship gate for the Gen 1-9 model: evaluates a checkpoint on train.py's held-out
test split (same stratified split, seed 42) in clean and sketch mode.

Run from the repo root:
    python training/gate.py --checkpoint models/best_sketch_v3.pth --data-dir data/raw

Exit 0 = PASS (ship), 1 = FAIL (stop and review; nothing ships).
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, os.path.dirname(__file__))
import train  # noqa: E402
from data_report import count_images  # noqa: E402

BARS = {"clean_all": 95.0, "sketch_all": 90.0, "clean_gen13": 95.7, "sketch_gen13": 92.8}
GEN13_COUNT = 386  # dex #1-386 = the classes the site already served
LIVE_THRESHOLD = 0.80  # Live search mode's confidence cut-off


def decide(metrics):
    lines, passed = [], True
    for key, bar in BARS.items():
        ok = metrics[key] >= bar
        passed &= ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} {key:<13} {metrics[key]:6.2f}%  (bar {bar:.1f}%)")
    lines.append("GATE: PASS" if passed else "GATE: FAIL")
    return passed, lines


def confidence_summary(conf_correct):
    conf = np.asarray(conf_correct, dtype=float)
    return {
        "share_ge_80": round(float((conf >= LIVE_THRESHOLD).mean() * 100), 2),
        "p10": float(np.percentile(conf, 10)),
    }


@torch.no_grad()
def evaluate(model, dataset, indices, device, batch):
    """Return per-sample (target, predicted, confidence) on `indices`."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch, shuffle=False, num_workers=4)
    targets, preds, confs = [], [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probs = torch.softmax(model(x.to(device)).float(), dim=1)
        c, p = probs.max(dim=1)
        targets += y.tolist(); preds += p.cpu().tolist(); confs += c.cpu().tolist()
    return np.array(targets), np.array(preds), np.array(confs)


def accuracy(targets, preds, mask=None):
    if mask is not None:
        targets, preds = targets[mask], preds[mask]
    return float((targets == preds).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description="Gen 1-9 ship gate")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--classes", default="models/class_names_v2.txt")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, clean_tf = train.build_transforms(224, sketch=False)
    _, sketch_tf = train.build_transforms(224, sketch=True)
    clean_ds = datasets.ImageFolder(args.data_dir, transform=clean_tf)
    sketch_ds = datasets.ImageFolder(args.data_dir, transform=sketch_tf)
    _, _, test_idx = train.stratified_split(clean_ds.targets, 0.1, 0.1, 42)

    model = train.build_model("convnext_tiny", len(clean_ds.classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device).eval()

    with open(args.classes, encoding="utf-8") as f:
        dex = [line.strip() for line in f if line.strip()]
    gen13 = {clean_ds.class_to_idx[n] for n in dex[:GEN13_COUNT]}

    random.seed(42); torch.manual_seed(42)  # sketch styles draw random parameters
    t, p, conf = evaluate(model, clean_ds, test_idx, device, args.batch)
    st, sp, _ = evaluate(model, sketch_ds, test_idx, device, args.batch)
    is13 = np.isin(t, list(gen13))
    metrics = {
        "clean_all": accuracy(t, p), "sketch_all": accuracy(st, sp),
        "clean_gen13": accuracy(t, p, is13), "sketch_gen13": accuracy(st, sp, is13),
    }
    passed, lines = decide(metrics)
    print(f"Test images: {len(test_idx)} ({int(is13.sum())} Gen 1-3) across {len(clean_ds.classes)} classes")
    print("\n".join(lines))

    per_class = []
    for c, name in enumerate(clean_ds.classes):
        m = t == c
        if m.any():
            per_class.append((accuracy(t, p, m), int(m.sum()), name))
    print("\nWorst 20 classes (clean):")
    for acc, n, name in sorted(per_class)[:20]:
        print(f"  {acc:6.1f}%  n={n:<3} {name}")

    counts = count_images(args.data_dir, dex)
    thin = [f"{n} ({counts[n]})" for n in dex if counts[n] < 10]
    print("\nThin classes (< 10 images): " + (", ".join(thin) if thin else "none"))

    s = confidence_summary(conf[t == p])
    print(f"\nLive check: {s['share_ge_80']:.1f}% of correct clean answers are >= 80% confident; "
          f"10th percentile confidence {s['p10'] * 100:.1f}%")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
