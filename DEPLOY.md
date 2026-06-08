# Deploying PokeGuess to Hugging Face Spaces

The app is a PyTorch + FastAPI server, so it needs a real container — not a
serverless host like Vercel. Hugging Face **Spaces (Docker SDK)** is the easiest
free option: it runs the `Dockerfile` in this repo and hosts the large model
files natively via Git LFS.

The app serves on port **7860** (the Spaces default) — already set in the
Dockerfile.

## One-time setup

1. **Create a Space**: https://huggingface.co/new-space
   - Owner: your account
   - Space name: `pokeguess`
   - License: MIT
   - **SDK: Docker** → *Blank*
   - Hardware: **CPU basic** (free) is enough

2. **Add the Space header.** A Spaces repo needs a `README.md` whose top has a
   YAML block telling HF how to run it. Create that file in the Space (or rename
   this project's README and prepend it). Minimum header:

   ```
   ---
   title: PokeGuess
   emoji: 🔍
   colorFrom: yellow
   colorTo: red
   sdk: docker
   app_port: 7860
   ---
   ```

3. **Push the code + models to the Space.** From this project folder:

   ```bash
   # Authenticate once (needs your HF token: https://huggingface.co/settings/tokens)
   pip install huggingface_hub
   huggingface-cli login

   # Add the Space as a second remote (replace <user>)
   git remote add space https://huggingface.co/spaces/<user>/pokeguess

   # The models are gitignored for GitHub (too big). Force-add them for the
   # Space only, tracked via Git LFS (HF requires LFS for files >10MB).
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   # best_sketch.pth is gitignored (too big for GitHub); the rest live in models/
   git add -f models/best_sketch.pth
   git add models/best_sketch_classes.txt models/fine_tuned_with_pencil_sketch.pth \
           models/class_names.txt models/class_names_v2.txt
   git commit -m "Add model weights for Space deploy"

   # Push this branch to the Space's main
   git push space HEAD:main
   ```

   The Space will build the Docker image and go live at
   `https://huggingface.co/spaces/<user>/pokeguess`.

## What gets served

- `app.py` loads **both** models (new ConvNeXt Gen 1+2 + legacy ResNet Gen 1)
  on CPU and auto-routes by confidence.
- `.dockerignore` keeps training/scraping code and the dataset out of the image,
  so it stays small.

## Notes / gotchas

- **First request is slow** on CPU basic while the models load into memory; it's
  fast afterward. Spaces may sleep on inactivity (free tier) and cold-start again.
- **Model size**: `best_sketch.pth` (~112 MB) and the legacy model (~45 MB) go up
  via Git LFS — fine on HF (unlike GitHub's 100 MB cap).
- To reduce memory, you can serve only the new model: in `app.py`, drop the
  `legacy_clf` load and return `new_clf.predict(...)` directly.
- The cry audio is `.ogg` (plays in Chrome/Firefox/Edge).
