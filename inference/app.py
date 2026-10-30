"""Flask inference service for PokeGuess. Internal API called by the ASP.NET web app.

Local:  flask --app inference/app.py run          (needs INFERENCE_API_KEY)
Azure:  gunicorn wsgi:app
"""
import hmac
import io
import os

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from model import OnnxClassifier, load_class_names
from preprocess import HALF_NORM, IMAGENET_NORM
from routing import route

MAX_ASPECT = 10  # longest side / shortest side; real photos, art and sketches are far below this


def create_app(new_clf=None, legacy_clf=None, gen1_names=None, api_key=None,
               legacy_trust_pct=None):
    model_dir = os.environ.get("MODEL_DIR", "models")
    path = lambda name: os.path.join(model_dir, name)  # noqa: E731

    if new_clf is None:
        new_clf = OnnxClassifier(path("convnext_gen1-9.onnx"),
                                 load_class_names(path("best_sketch_classes.txt")), IMAGENET_NORM)
    if legacy_clf is None:
        legacy_clf = OnnxClassifier(path("resnet18_gen1.onnx"),
                                    load_class_names(path("class_names.txt")), HALF_NORM)
    if gen1_names is None:
        gen1_names = set(load_class_names(path("class_names_v2.txt"))[:151])
    if api_key is None:
        api_key = os.environ.get("INFERENCE_API_KEY")
        if not api_key:
            raise RuntimeError("INFERENCE_API_KEY is not set")
    if legacy_trust_pct is None:
        legacy_trust_pct = float(os.environ.get("LEGACY_TRUST", "0.55")) * 100

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict():
        given = request.headers.get("X-Api-Key", "")
        if not hmac.compare_digest(given.encode(), api_key.encode()):
            return jsonify(error="unauthorized"), 401

        upload = request.files.get("file")
        try:
            image = Image.open(io.BytesIO(upload.read()))
            image.load()
        except (AttributeError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="Could not read image file."), 400
        # Resize scales the short side to 256, so a sliver image would explode in memory.
        if max(image.size) > MAX_ASPECT * min(image.size):
            return jsonify(error="Could not read image file."), 400

        top_k = min(max(request.args.get("top_k", 3, type=int), 1), 10)
        preds, model = route(new_clf.predict(image, top_k),
                             lambda: legacy_clf.predict(image, top_k),
                             gen1_names, legacy_trust_pct)
        return {
            "model": model,
            "predictions": [{"name": n, "confidence": round(c, 2)} for n, c in preds],
        }

    return app
