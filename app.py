"""FastAPI web app for PokeGuess.

Serves a single static page (static/index.html) with drag-drop + clipboard
paste, and a JSON prediction endpoint backed by the shared PokemonClassifier.

Run with:
    uvicorn app:app --reload
then open http://127.0.0.1:8000
"""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from classifier import (PokemonClassifier, load_class_names,
                        IMAGENET_NORM, HALF_NORM)

# --- New model: ConvNeXt-Tiny, Gen 1+2 (251 classes) ----------------------- #
NEW_MODEL_PATH = "models/best_sketch.pth"
NEW_CLASSES = load_class_names("models/best_sketch_classes.txt")

# --- Legacy model: ResNet-18, Gen 1 only, trained on a larger dataset ------ #
LEGACY_MODEL_PATH = "models/fine_tuned_with_pencil_sketch.pth"
LEGACY_CLASSES = load_class_names("models/class_names.txt")

# Confidence above which we trust the legacy Gen-1 model (when the new model
# already agrees the input is a Gen-1 Pokemon). Tunable via env.
LEGACY_TRUST = float(os.environ.get("LEGACY_TRUST", "0.55")) * 100

# Gen-1 = National Dex #1..151 = first 151 entries of the canonical dex-order
# list. Used to decide whether to consult the legacy model at all.
_dex = load_class_names("models/class_names_v2.txt")
GEN1_NAMES = set(_dex[:151])

# Legacy class list has messy/duplicate spellings; map them to canonical names
# so sprites/cries and display stay consistent with the new model.
LEGACY_FIX = {
    "Farfetchd": "Farfetch'd", "Mime": "Mr. Mime", "MrMime": "Mr. Mime",
    "Nidoran1": "Nidoran-F", "Nidoran2": "Nidoran-M",
}

app = FastAPI(title="PokeGuess", description="Pokémon classifier (Gen 1+2)")

new_clf = PokemonClassifier(NEW_MODEL_PATH, NEW_CLASSES,
                            backbone="convnext_tiny", norm=IMAGENET_NORM)
legacy_clf = PokemonClassifier(LEGACY_MODEL_PATH, LEGACY_CLASSES,
                               backbone="resnet18", norm=HALF_NORM)


def _route(image, top_k):
    """Auto-route: the new (Gen 1+2) model decides the generation; if it says
    Gen 1 and the legacy model is confident, defer to legacy (more training
    data on Gen 1). Otherwise use the new model."""
    new_preds = new_clf.predict(image, top_k=top_k)
    new_top = new_preds[0][0]

    if new_top in GEN1_NAMES:
        legacy_preds = legacy_clf.predict(image, top_k=top_k)
        legacy_preds = [(LEGACY_FIX.get(n, n), c) for n, c in legacy_preds]
        if legacy_preds[0][1] >= LEGACY_TRUST:
            return legacy_preds, "legacy-gen1"
    return new_preds, "gen1-3"


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read image file.")

    results, source = _route(image, top_k)
    return {
        "model": source,
        "predictions": [
            {"name": name, "confidence": round(conf, 2)} for name, conf in results
        ]
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
