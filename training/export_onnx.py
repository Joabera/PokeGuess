"""Export the PyTorch checkpoints to ONNX for the Flask inference service,
then check that ONNX (with inference/preprocess.py) matches PyTorch (with the
torchvision transform it was trained with).

Run from the repo root:
    python training/export_onnx.py
"""
import sys

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, "inference")
from preprocess import HALF_NORM, IMAGENET_NORM, to_tensor  # noqa: E402

EXPORTS = [
    ("convnext_tiny", "models/best_sketch.pth", "models/best_sketch_classes.txt",
     IMAGENET_NORM, "models/convnext_gen1-3.onnx"),
    ("resnet18", "models/fine_tuned_with_pencil_sketch.pth", "models/class_names.txt",
     HALF_NORM, "models/resnet18_gen1.onnx"),
]
INT8_OUT = "models/convnext_gen1-3_int8.onnx"  # CPU fallback for Live mode in the browser


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


def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def check_batch_and_int8(name, fp32_path, int8_path, norm):
    """Live mode scores several crops per call (batch) and falls back to int8 on CPU."""
    tensors = [to_tensor(img, *norm) for img in sample_images()]
    fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    singles = np.concatenate([fp32.run(None, {"input": t})[0] for t in tensors])
    try:
        batched = fp32.run(None, {"input": np.concatenate(tensors)})[0]
    except Exception as e:  # fixed-batch export rejects n > 1
        raise SystemExit(f"PARITY FAIL {name}: batch ({e.__class__.__name__})")
    if not np.allclose(batched, singles, atol=1e-3):
        raise SystemExit(f"PARITY FAIL {name}: batch")
    int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    quant = int8.run(None, {"input": np.concatenate(tensors)})[0]
    # Compare probabilities, not argmax: the parity images aren't Pokémon, so the model's
    # top classes are near-ties (~1.5%) that any quantization noise can reorder.
    drift = np.abs(softmax(quant) - softmax(singles)).max()
    if drift > 0.02:
        raise SystemExit(f"PARITY FAIL {name}: int8 (max probability drift {drift:.3f})")
    print(f"PARITY OK {name} (batch, int8)")


def main():
    for backbone, pth, classes, norm, out in EXPORTS:
        with open(classes, encoding="utf-8") as f:
            num_classes = sum(1 for line in f if line.strip())
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
            check_batch_and_int8(name, out, INT8_OUT, norm)


if __name__ == "__main__":
    main()
