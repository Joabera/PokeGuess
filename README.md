# PokeGuess — Pokémon Image Classifier (Gen 1–3)

**Live:** https://pokeguess-web-773dd.azurewebsites.net · [Live camera](https://pokeguess-web-773dd.azurewebsites.net/Live) · [Stats](https://pokeguess-web-773dd.azurewebsites.net/Stats)

Identifies **the 386 Pokémon of Generations 1–3** from a photo, illustration, or hand-drawn sketch. You drop, paste, or snap an image. PokeGuess shows its top 3 guesses, the official sprite, and the Pokémon's cry, then asks whether it got it right. That feedback feeds a live accuracy dashboard.

**Live mode** points your webcam or phone camera at a card, picture, or drawing and recognizes Pokémon in real time, drawing green boxes over the video. The model runs **in your browser** (ONNX Runtime Web), so nothing is uploaded per frame; **Snap** sends just the boxed crop through the same pipeline as uploads, tagged as `live` in Stats.

An **ASP.NET Core MVC** web app (C#, jQuery, EF Core + Azure SQL) sits in front of a **Flask** inference microservice running two **ONNX** models. Both are deployed to **Azure App Service** by GitHub Actions.

> Originally a university project (a ResNet-18 Gen-1 classifier), then rebuilt with a self-collected dataset and ConvNeXt, and now re-platformed onto .NET + Azure.

## Architecture

```
Browser (MVC views + jQuery)
   │  AJAX: POST /api/guess (image)        POST /api/guess/{id}/feedback
   ▼
ASP.NET Core MVC  (C#, .NET 10) ── Azure App Service
   │  HttpClient → POST /predict (multipart, X-Api-Key)     EF Core
   ▼                                                        ▼
Flask inference (Python) ── Azure App Service              Azure SQL (SQLite locally)
   ONNX Runtime: ConvNeXt-Tiny (Gen 1–3) + ResNet-18 (Gen 1)
```

- **The browser only talks to the MVC app.** MVC calls Flask server-side with a shared API key, so there's no CORS and no public model API.
- **Two-model routing:** ConvNeXt decides the generation. For Gen 1 it defers to the original ResNet-18 (trained on more Gen 1 data) when that model is ≥ 55% confident. The UI shows which model answered.
- **Feedback loop:** every guess is stored (never the image). ✓/✗ feedback powers `/Stats`: accuracy overall, by model, by generation, and by source (upload vs live), plus the most common mix-ups.
- **Live mode (`/Live`):** ConvNeXt runs in the browser on **WebGPU** (fp32, 113 MB; ~40 fps with the guide box, several fps searching the whole frame) or falls back to the **CPU** (int8, 28 MB; guide box only). The model downloads once per device and is kept in the browser's Cache API. Search mode scores a grid of crops per frame, keeps confident hits that hold for 2 of 3 frames, then shrinks each box while the same Pokémon stays ≥ 80%. The web app serves the model files from `/models/` with immutable caching.

## What's where

| Area | Tech | Path |
|---|---|---|
| Web app, pages, REST API | ASP.NET Core MVC (C#), Razor, Bootstrap 5, jQuery AJAX | `src/PokeGuess.Web` |
| Data | EF Core, Azure SQL / SQLite, aggregate queries | `src/PokeGuess.Web/Data`, `Services/StatsService.cs` |
| Stats API | `GET /api/stats`, JSON by default, **XML** with `Accept: application/xml` | `Controllers/StatsController.cs` |
| Inference service | Python, Flask, ONNX Runtime, gunicorn | `inference/` |
| Live mode | ONNX Runtime Web (WebGPU / WASM), ES modules, Cache API | `Views/Live`, `wwwroot/js/live/` |
| Tests | xUnit + `WebApplicationFactory`, pytest, `node --test` | `src/PokeGuess.Web.Tests` (incl. `js/`), `inference/tests` |
| Training + export | PyTorch, torchvision, ONNX parity check | `training/` |
| Infra + CI/CD | az CLI, GitHub Actions, OIDC login | `infra/setup.sh`, `.github/workflows` |

Hardening: 5 MB upload cap, a per-IP rate limit on guesses (10/min, `X-Forwarded-For`-aware behind Azure's front end), atomic one-time feedback (conditional `UPDATE`), and graceful 503 handling while the inference service wakes up. Mobile: responsive layout, camera capture, and client-side downscaling of large photos before upload.

## Run locally

Needs the .NET 10 SDK and Python 3.12+.

```bash
# 1. Models (ONNX, published as a GitHub release; ~190 MB). models/ feeds the Live page locally.
gh release download models-v2 -p "*.onnx" -D models
cp models/convnext_gen1-3.onnx models/resnet18_gen1.onnx inference/models/
cp models/best_sketch_classes.txt models/class_names.txt models/class_names_v2.txt inference/models/

# 2. Inference service → http://127.0.0.1:5000
cd inference
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # macOS/Linux: .venv/bin/pip
INFERENCE_API_KEY=dev MODEL_DIR=models .venv/Scripts/flask --app app.py run
cd ..

# 3. Web app (SQLite, created automatically) → http://localhost:5252
dotnet user-secrets set Inference:ApiKey dev --project src/PokeGuess.Web
dotnet run --project src/PokeGuess.Web
```

Tests: `dotnet test`, `node --test "src/PokeGuess.Web.Tests/js/*.test.mjs"` (Node 22), and `cd inference && pytest`.

Local SQLite is created on first run but never altered afterwards. After a schema change (e.g. the `Source` column), delete `src/PokeGuess.Web/pokeguess.db*` and restart. Azure SQL uses EF migrations instead.

## Deploy to Azure

```bash
az login && gh auth login

# Once: publish the ONNX models the deploy downloads (run training/export_onnx.py first)
gh release create models-v2 models/convnext_gen1-3.onnx models/resnet18_gen1.onnx models/convnext_gen1-3_int8.onnx \
  --title "Model weights v2" --notes "Dynamic-batch ConvNeXt + int8 for Live mode; ResNet-18 unchanged"

bash infra/setup.sh        # creates the resource group, B1 plan, 2 web apps, Azure SQL free offer, OIDC identity, repo variables
git push origin main       # the Deploy workflow tests, then ships both apps
```

If `setup.sh` stops partway, re-run it with the suffix it printed (`SUFFIX=abcde bash infra/setup.sh`) so it reuses the same resources. `bash infra/test_setup.sh` dry-runs the script against stubbed `az`/`gh`.

Cost is about $13/month for the B1 plan, which hosts both apps; the SQL database is on the free offer. Tear down with `az group delete -n pokeguess-rg --yes`.

## Training & export

The dataset comes from official sources (PokéAPI sprites + cropped TCG card art, about 13k images):

```bash
pip install -r training/requirements.txt
python training/build_class_list.py     # -> models/class_names_v2.txt, models/pokeapi_slugs.json
python training/scrape_dataset.py all   # seed + cards + dedup + CLIP/NSFW filter -> data/raw/<Name>/

# Phase 1 — color baseline; Phase 2 — randomized sketch fine-tune
python training/train.py --data-dir data/raw --epochs 30 --out models/best_color.pth --out-dir runs/color
python training/train.py --data-dir data/raw --epochs 15 --sketch --init-weights models/best_color.pth \
    --lr 5e-5 --out models/best_sketch.pth --out-dir runs/sketch

# Export both checkpoints to ONNX; fails unless ONNX + preprocessing match PyTorch
python training/export_onnx.py
```

`.pth` / `.onnx` weights and the dataset are not committed (size / licensing).

## Results

| Model | Classes | Test accuracy |
|---|---|---|
| Legacy ResNet-18 | 150 (Gen 1) | — (original project) |
| ConvNeXt-Tiny, color | 386 (Gen 1–3) | **98.7%** |
| ConvNeXt-Tiny, sketch fine-tune | 386 (Gen 1–3) | **95.8%** (sketch-style eval) |

The sketch fine-tune trades a few points of clean-image accuracy for robustness on hand-drawn inputs, which is what a "guess the sketch" demo wants. Real-world accuracy from user feedback is on the live `/Stats` page.

## License

[MIT](LICENSE)
