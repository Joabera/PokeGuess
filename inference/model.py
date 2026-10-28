"""ONNX Runtime classifier for one exported model."""
import numpy as np
import onnxruntime as ort

from preprocess import to_tensor


def load_class_names(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class OnnxClassifier:
    def __init__(self, model_path, class_names, norm):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.class_names = class_names
        self.norm = norm

    def predict(self, image, top_k):
        """Return [(class_name, confidence_pct)] for the top_k classes, best first."""
        logits = self.session.run(None, {"input": to_tensor(image, *self.norm)})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        top = np.argsort(probs)[::-1][:top_k]
        return [(self.class_names[i], float(probs[i]) * 100) for i in top]
