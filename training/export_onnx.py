"""Export the PyTorch checkpoints to ONNX for the Flask inference service,
then check that ONNX (with inference/preprocess.py) matches PyTorch (with the
torchvision transform it was trained with).

Run from the repo root:
    python training/export_onnx.py
"""
import os
import random
import shutil
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

sys.path.insert(0, "inference")
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import HALF_NORM, IMAGENET_NORM, to_tensor  # noqa: E402
import train  # noqa: E402  (same test split as training and the gate)

CLASS_LIST = "models/class_names_v2.txt"
DATA_DIR = "data/raw"
INT8_SAMPLES = 500
INT8_MIN_AGREEMENT = 98.0

EXPORTS = [
    ("convnext_tiny", "models/best_sketch_v3.pth", "models/best_sketch_v3_classes.txt",
     IMAGENET_NORM, "models/convnext_gen1-9.onnx"),
    # The Gen 1 specialist is exported unchanged; it is never retrained.
    ("resnet18", "models/fine_tuned_with_pencil_sketch.pth", "models/class_names.txt",
     HALF_NORM, "models/resnet18_gen1.onnx"),
]
INT8_OUT = "models/convnext_gen1-9_int8.onnx"  # CPU fallback for Live mode in the browser
SERVED_CLASSES = "models/best_sketch_classes.txt"  # what the inference service and web app read


def build_backbone(backbone, num_classes):
    if backbone == "convnext_tiny":
        m = models.convnext_tiny()
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    else:
        m = models.resnet18()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def sample_images():
    gradient = np.zeros((480, 640, 3), dtype=np.uint8)
    gradient[..., 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    gradient[..., 1] = np.linspace(0, 255, 480, dtype=np.uint8)[:, None]
    rgba = np.random.default_rng(0).integers(0, 256, (500, 300, 4), dtype=np.uint8)
    gray = np.random.default_rng(1).integers(0, 256, (1000, 1000), dtype=np.uint8)
    return [Image.fromarray(gradient, "RGB"), Image.fromarray(rgba, "RGBA"),
            Image.fromarray(gray, "L")]


def check_parity(name, model, onnx_path, norm):
    reference = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(*norm),
    ])
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for img in sample_images():
        ours = to_tensor(img, *norm)
        theirs = reference(img.convert("RGB")).unsqueeze(0).numpy()
        if not np.allclose(ours, theirs, atol=1e-4):
            raise SystemExit(f"PARITY FAIL {name}: preprocessing differs "
                             f"(max {np.abs(ours - theirs).max():.2e})")
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(theirs)).numpy()
        onnx_logits = session.run(None, {"input": ours})[0]
        if not np.allclose(onnx_logits, torch_logits, atol=1e-3) or \
                onnx_logits.argmax() != torch_logits.argmax():
            raise SystemExit(f"PARITY FAIL {name}: logits differ "
                             f"(max {np.abs(onnx_logits - torch_logits).max():.2e})")
    print(f"PARITY OK {name}")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def check_classes(classes_path):
    """The model's labels must be exactly the canonical class list, or the app shows wrong names."""
    model_classes, dex = read_lines(classes_path), read_lines(CLASS_LIST)
    if len(model_classes) != len(dex) or set(model_classes) != set(dex):
        missing = sorted(set(dex) - set(model_classes))[:5]
        extra = sorted(set(model_classes) - set(dex))[:5]
        raise SystemExit(f"PARITY FAIL classes: model has {len(model_classes)}, list has {len(dex)}; "
                         f"missing {missing}, unexpected {extra}")
    print(f"CLASSES OK ({len(dex)})")


def check_batch(name, fp32_path, norm):
    """Live mode scores several crops per call."""
    tensors = [to_tensor(img, *norm) for img in sample_images()]
    fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    singles = np.concatenate([fp32.run(None, {"input": t})[0] for t in tensors])
    try:
        batched = fp32.run(None, {"input": np.concatenate(tensors)})[0]
    except Exception as e:  # fixed-batch export rejects n > 1
        raise SystemExit(f"PARITY FAIL {name}: batch ({e.__class__.__name__})")
    if not np.allclose(batched, singles, atol=1e-3):
        raise SystemExit(f"PARITY FAIL {name}: batch")


def check_int8(name, fp32_path, int8_path, norm, n=INT8_SAMPLES, batch=50):
    """The CPU fallback (int8) must give the same top-1 as fp32 on real test images."""
    dataset = datasets.ImageFolder(DATA_DIR)
    _, _, test_idx = train.stratified_split(dataset.targets, 0.1, 0.1, 42)
    picks = random.Random(0).sample(test_idx, min(n, len(test_idx)))
    fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    agree = 0
    for start in range(0, len(picks), batch):
        x = np.concatenate([to_tensor(Image.open(dataset.samples[i][0]), *norm)
                            for i in picks[start:start + batch]])
        agree += int((fp32.run(None, {"input": x})[0].argmax(1) ==
                      int8.run(None, {"input": x})[0].argmax(1)).sum())
    pct = agree / len(picks) * 100
    print(f"int8 top-1 agreement with fp32: {pct:.1f}% on {len(picks)} test images")
    if pct < INT8_MIN_AGREEMENT:
        raise SystemExit(f"PARITY FAIL {name}: int8 agreement {pct:.1f}%")
    print(f"PARITY OK {name} (batch, int8)")


def main():
    check_classes(EXPORTS[0][2])
    for backbone, pth, classes, norm, out in EXPORTS:
        num_classes = len(read_lines(classes))
        model = build_backbone(backbone, num_classes)
        model.load_state_dict(torch.load(pth, map_location="cpu"))
        model.eval()
        # ConvNeXt gets a dynamic batch axis so the browser can score many crops per call.
        dynamic = {"input": {0: "n"}, "logits": {0: "n"}} if backbone == "convnext_tiny" else None
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), out,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes=dynamic, opset_version=17, dynamo=False)
        name = out.split("/")[-1].removesuffix(".onnx")
        check_parity(name, model, out, norm)
        if backbone == "convnext_tiny":
            quantize_dynamic(out, INT8_OUT, weight_type=QuantType.QUInt8)
            check_batch(name, out, norm)
            check_int8(name, out, INT8_OUT, norm)
    # Only now that every check passed does the served class order switch over.
    shutil.copyfile(EXPORTS[0][2], SERVED_CLASSES)
    print(f"Copied {EXPORTS[0][2]} -> {SERVED_CLASSES}")


if __name__ == "__main__":
    main()
