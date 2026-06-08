# PokeGuess — Pokémon Image Classifier (Gen 1 & 2)

### 👉 [**Try it live on Hugging Face Spaces**](https://huggingface.co/spaces/JTemotio/pokeguess)

[![Live Demo](https://img.shields.io/badge/🤗%20Spaces-Try%20it%20live-yellow)](https://huggingface.co/spaces/JTemotio/pokeguess)

A deep-learning image classifier that identifies **the 251 Pokémon of Generations 1 & 2** from a single image — photo, illustration, or hand-drawn sketch. Built with PyTorch, it pairs a modern **ConvNeXt-Tiny** model with a small **FastAPI web app** that shows the prediction, the official sprite, and even plays the Pokémon's cry.

> Originally a university project (a ResNet-18 Gen-1 classifier); since rebuilt with a self-collected dataset, a stronger backbone, Gen-2 coverage, and a browser UI.

---

## Highlights

- **251 classes, Gen 1 + 2** — trained on a dataset built from official sources (PokéAPI sprites + cropped Pokémon TCG card art), ~9k clean images.
- **ConvNeXt-Tiny** backbone (ImageNet pre-trained), re-headed for 251 classes — **96.7% test accuracy** on clean images, **93.9%** on sketch-style inputs.
- **Two-phase training:** color baseline → randomized **sketch** fine-tune (pencil / edge / threshold styles) so it generalizes to drawn inputs.
- **Two-model auto-routing:** the new Gen-1+2 model detects the generation; for Gen-1 Pokémon it can defer to the original ResNet-18 model (trained on a larger Gen-1 set) when that model is confident.
- **FastAPI web app:** drag-drop / paste / file-pick, top-3 confidence bars, the official **sprite + cry** from PokéAPI, and a playful "is this a Digimon?" response below 60% confidence.
- **Reproducible data agent:** `scrape_dataset.py` rebuilds the dataset end-to-end (seed → cards → dedup → CLIP/NSFW filter) with progress bars.

## Tech Stack

| Layer | Tools |
|---|---|
| Model | PyTorch, torchvision (ConvNeXt-Tiny, ResNet-18) |
| Training | CUDA + AMP, AdamW, cosine LR, weighted sampling, TensorBoard |
| Data / Image | Pillow, OpenCV, imagehash, OpenCLIP, icrawler |
| Web app | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/JS, PokéAPI (sprites + cries) |

---

## Project Structure

```
PokeGuess/
├── app.py                  # FastAPI app — serves the page + /predict (auto-routes 2 models)
├── classifier.py           # Shared inference (model build + preprocessing)
├── static/index.html       # Single-page UI: drag-drop/paste, sprite, cry
├── test_pokemon_model_gui.py   # Optional Tkinter desktop GUI
├── Dockerfile              # Container for Hugging Face Spaces (see DEPLOY.md)
├── requirements.txt        # Full deps (training + dataset tooling)
├── requirements-app.txt    # Slim runtime deps (serving only)
│
├── models/                 # Weights + class lists
│   ├── best_sketch.pth*        # New ConvNeXt model (sketch fine-tuned) — not in git (size)
│   ├── fine_tuned_with_pencil_sketch.pth   # Legacy ResNet-18 Gen-1 model
│   ├── best_sketch_classes.txt # 251 classes (ImageFolder order; used by the app)
│   ├── class_names.txt         # Legacy 154-class list
│   ├── class_names_v2.txt      # 251 classes in National Dex order
│   └── pokeapi_slugs.json      # Display-name → PokéAPI slug map
│
├── scripts/                # Data + training tooling
│   ├── train.py                # ConvNeXt-Tiny training, both phases (color + --sketch)
│   ├── sketch.py               # Randomized pencil/edge/threshold augmentation
│   ├── scrape_dataset.py       # Dataset-building agent (seed/cards/dedup/clip)
│   └── build_class_list.py     # Generates the canonical 251-class list from PokéAPI
│
├── legacy/                 # Original Gen-1 training scripts (kept for history)
├── docs/Final_Report.pdf   # Original project write-up
├── DEPLOY.md               # Hugging Face Spaces deployment guide
└── README.md
```

> Large weights (`best_*.pth`, ~112 MB) and the image dataset (`data/`) are **not** committed (size / licensing). Rebuild the dataset with `scripts/scrape_dataset.py` and retrain with `scripts/train.py`, or host the weights externally (see `DEPLOY.md`). Run scripts from the repo root, e.g. `python scripts/train.py`.

---

## Getting Started (web app)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app:app --reload      # then open http://127.0.0.1:8000
```

Drop, paste (Ctrl/Cmd+V), or pick an image. You get the top-3 guesses with confidence bars; on a confident match it shows the official sprite and plays the cry. A small badge notes which model answered (Gen 1+2 vs. the Gen-1 legacy model).

> The app needs the model files under `models/` (`best_sketch.pth` + `best_sketch_classes.txt` for the new model, `fine_tuned_with_pencil_sketch.pth` + `class_names.txt` for the legacy one). The cry audio is `.ogg` — plays in Chrome/Firefox/Edge.

---

## Rebuilding the dataset

The dataset is assembled from official, correctly-labeled sources (no random web scraping in the default flow):

```bash
python scripts/build_class_list.py    # -> models/class_names_v2.txt, models/pokeapi_slugs.json
python scripts/scrape_dataset.py all  # seed (PokéAPI) + cards (TCG) + dedup + CLIP/NSFW filter
```

Output is an `ImageFolder` layout in `data/raw/<Name>/`. Each stage is resumable and shows a progress bar. An opt-in `scrape` stage (Bing/DuckDuckGo) exists but is **not** part of `all` — it pulled too much noise.

## Training

```bash
# Phase 1 — color baseline
python scripts/train.py --data-dir data/raw --epochs 30 \
    --out models/best_color.pth --out-dir runs/color

# Phase 2 — randomized sketch fine-tune (warm-start from phase 1)
python scripts/train.py --data-dir data/raw --epochs 15 --sketch \
    --init-weights models/best_color.pth --lr 5e-5 \
    --out models/best_sketch.pth --out-dir runs/sketch
```

Features: stratified train/val/test split, weighted sampling for class imbalance, mixed precision, early stopping, TensorBoard logs, and a confusion-matrix + per-class accuracy report written to the run dir. GPU is used automatically when available (CUDA build of torch).

---

## Results

| Model | Classes | Test accuracy |
|---|---|---|
| Legacy ResNet-18 | 150 (Gen 1) | — (original project) |
| ConvNeXt-Tiny, color | 251 (Gen 1+2) | **96.7%** |
| ConvNeXt-Tiny, sketch fine-tune | 251 (Gen 1+2) | **93.9%** (sketch-style eval) |

The sketch fine-tune trades a few points of clean-image accuracy for robustness on hand-drawn inputs — the intended behavior for a "guess the sketch" demo.

---

## License

[MIT](LICENSE) — free to use, modify, and learn from.
