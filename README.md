# PokeGuess — Pokémon Image Classifier

A deep-learning image classifier that identifies **150 first-generation Pokémon** from a single image. Built with PyTorch using transfer learning on ResNet-18, with a two-phase training pipeline (baseline + stylized fine-tuning) and a Tkinter desktop GUI for live predictions.

> A university project I built before the recent wave of AI tools — included here as an end-to-end example of training, fine-tuning, and deploying a custom vision model.

---

## Highlights

- **Transfer learning** on a ResNet-18 backbone (ImageNet pre-trained) re-headed for 150 Pokémon classes.
- **Two-phase training pipeline:**
  1. *Baseline* — train on color photos with standard augmentations (random crop, horizontal flip, normalization).
  2. *Stylized fine-tune* — adapt the baseline checkpoint to a **pencil-sketch domain** (grayscale → invert → Gaussian blur → blend), letting the model generalize to drawn/sketched inputs rather than only screenshots.
- **Cosine/StepLR schedule** with checkpointing of the best validation accuracy each epoch.
- **Tkinter GUI** that classifies images from disk *or* directly from the clipboard (paste a screenshot, get a prediction + softmax confidence).
- Self-contained: a single trained checkpoint (`fine_tuned_with_pencil_sketch.pth`) is included so the GUI can be run without retraining.

## Tech Stack

| Layer | Tools |
|---|---|
| Model | PyTorch, torchvision (ResNet-18) |
| Data / Image | Pillow, OpenCV |
| UI | Tkinter |
| Training | CUDA (optional), Adam + StepLR |

---

## Project Structure

```
PokeGuess/
├── train_baseline_model.py            # Phase 1: train ResNet-18 on color photos
├── fine_tune_with_color_variation.py  # Phase 2: fine-tune with pencil-sketch augmentation
├── test_pokemon_model_gui.py          # Tkinter GUI for live inference
├── fine_tuned_with_pencil_sketch.pth  # Trained model weights (~45 MB)
├── requirements.txt
├── Final_Report.pdf                   # Full write-up: methodology, experiments, results
└── README.md
```

> The training dataset (~150 folders of Pokémon images) is **not included** in this repo due to size and licensing. See *Dataset* below for the expected layout.

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> GPU training is automatic if a CUDA-capable PyTorch build is installed; otherwise it falls back to CPU.

### 2. Run the GUI (uses included weights)

```bash
python test_pokemon_model_gui.py
```

You'll get a small window with two buttons:
- **Select Image** — pick a `.jpg` / `.png` / `.bmp` from disk.
- **Paste Image** — paste whatever is on your clipboard (great for testing with screenshots).

The prediction and softmax confidence are shown below the image.

> You'll also need `class_names.txt` (one class per line, in the same alphabetical order produced by `torchvision.datasets.ImageFolder`) in the working directory. It is written automatically the first time you run training.

### 3. Train from scratch (optional)

```bash
# Phase 1 — baseline
python train_baseline_model.py

# Phase 2 — fine-tune with pencil-sketch augmentation
python fine_tune_with_color_variation.py
```

Each script saves the best validation checkpoint to `.pth` and emits a timestamped final model.

---

## Dataset

Organize your images as one folder per class:

```
Pokemons/
├── Bulbasaur/
│   ├── img_0001.jpg
│   └── ...
├── Charmander/
│   └── ...
└── ... (150 folders)
```

Then update the `data_folder` path at the bottom of each training script.

---

## Approach & Results

The full methodology, experiments, and final metrics are in **[Final_Report.pdf](Final_Report.pdf)**. In short:

- ResNet-18 was chosen for a strong accuracy/parameter trade-off on a modest dataset.
- The pencil-sketch fine-tune was motivated by the observation that the baseline overfit to clean illustrated artwork and failed on hand-drawn or low-contrast inputs. Training on stylized variants improved robustness on out-of-distribution images at a small cost in raw top-1 accuracy.
- The classifier is trained on the 150 original Pokémon, with `class_names.txt` mapping output indices back to names.

## What I'd do differently today

This was an early ML project of mine — keeping it here as an honest snapshot. Things I'd change with more experience:

- Replace the global `data_folder` hardcoded path with CLI args (`argparse`) or a small config file.
- Track experiments with **Weights & Biases** or **TensorBoard** instead of stdout prints.
- Use **stratified train/val splits** rather than `random_split` to guarantee per-class coverage.
- Swap Tkinter for a small **FastAPI + web frontend** so the demo runs in the browser.
- Move the pencil-sketch transform to a `torchvision.transforms.v2`-compatible class for performance.

---

## License

[MIT](LICENSE) — free to use, modify, and learn from.
