"""Per-class image counts for the training set, checked against the class file.

Run from the repo root:
    python training/data_report.py [--data-dir data/raw]

Exits 1 if any class in models/class_names_v2.txt has no folder or no images
(e.g. the card stage ran with a stale cache and a new class got nothing).
"""
import argparse
import os
import sys

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
GEN_LAST_DEX = [151, 251, 386, 493, 649, 721, 809, 905, 1025]


def count_images(data_dir, names):
    counts = {}
    for name in names:
        folder = os.path.join(data_dir, name)
        counts[name] = sum(
            1 for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXT
        ) if os.path.isdir(folder) else 0
    return counts


def report(counts, thin=10):
    lines = [f"{len(counts)} classes, {sum(counts.values())} images"]
    names = list(counts)
    start = 0
    for gen, last in enumerate(GEN_LAST_DEX, start=1):
        chunk = names[start:last]
        if chunk:
            lines.append(f"  Gen {gen}: {len(chunk)} classes, {sum(counts[n] for n in chunk)} images")
        start = last
    low = [n for n in names if 0 < counts[n] < thin]
    if low:
        lines.append(f"Thin classes (< {thin} images): " + ", ".join(f"{n} ({counts[n]})" for n in low))
    empty = [n for n in names if counts[n] == 0]
    if empty:
        lines.append("MISSING/EMPTY classes: " + ", ".join(empty))
    return (1 if empty else 0), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--classes", default="models/class_names_v2.txt")
    args = ap.parse_args()
    with open(args.classes, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    code, lines = report(count_images(args.data_dir, names))
    print("\n".join(lines))
    sys.exit(code)


if __name__ == "__main__":
    main()
