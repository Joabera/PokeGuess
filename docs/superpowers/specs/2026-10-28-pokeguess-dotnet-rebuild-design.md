# PokeGuess .NET Rebuild — Design

**Date:** 2026-10-28
**Branch:** `feat/dotnet-rebuild`
**Status:** Approved in brainstorming, pending spec review

## Goal

Rebuild the PokeGuess serving stack as a portfolio project aimed at a role asking for:
C#/ASP.NET web apps, Python/Flask, Git, Azure App Service deployment, and basic SQL,
HTML, JavaScript, jQuery, JSON, XML, AJAX, REST, plus responsive (desktop + mobile) UI.

**Success criteria**

- Live on Azure App Service, deployed by GitHub Actions on merge to `main`.
- Same core experience as today (upload/paste → top-3 → sprite + cry → low-confidence joke), working well on phones.
- Users can mark a guess correct/wrong; a Stats page shows real accuracy from that feedback.
- Each job requirement is visibly exercised by the code (see mapping below).

**Explicitly out of scope**

- SSO / user accounts (decided: not a fit for a public guessing app).
- Retraining or changing the models. Training stays in Python/PyTorch.
- Storing uploaded images.
- Docker, Hugging Face Spaces deployment (removed from the repo; stays in git history).

## Requirement mapping

| Requirement | Where it lives |
|---|---|
| C# / ASP.NET web apps | `src/PokeGuess.Web` — ASP.NET Core MVC, .NET 10 |
| Python / Flask | `inference/` — Flask inference service |
| Git | Feature branches + PRs |
| Azure App Service | Two web apps on one B1 Linux plan, deployed via GitHub Actions |
| SQL | EF Core + Azure SQL (SQLite locally); stats via aggregate queries |
| HTML / JS / jQuery / AJAX / JSON / REST | MVC views; jQuery `$.ajax` to `/api/*` JSON endpoints; MVC → Flask REST |
| XML | `GET /api/stats` content-negotiates JSON or XML |
| Multiple form factors | Bootstrap 5 responsive layout; `capture="environment"` camera input on mobile |

## Architecture

```
Browser (MVC views + jQuery)
   │  AJAX: POST /api/guess (image)        POST /api/guess/{id}/feedback
   ▼
ASP.NET Core MVC  (C#, .NET 10) ── Azure App Service #1
   │  HttpClient → POST /predict (multipart, X-Api-Key)     EF Core
   ▼                                                        ▼
Flask inference (Python) ── App Service #2                 Azure SQL (SQLite locally)
   ONNX Runtime: ConvNeXt-Tiny (Gen 1–3) + ResNet-18 (Gen 1)
```

- The browser only talks to the MVC app. MVC calls Flask server-side, so there is no CORS and Flask is not a public API.
- PokéAPI (sprite + cry) stays a client-side call from the browser.

## Repo layout

```
src/PokeGuess.Web/          ASP.NET Core MVC app
src/PokeGuess.Web.Tests/    xUnit tests
inference/                  Flask app, routing, ONNX loader, requirements.txt, tests/
training/                   current scripts/ + legacy/ moved here, plus export_onnx.py
infra/setup.sh              az CLI script that creates all Azure resources
.github/workflows/          ci.yml (PR) and deploy.yml (main)
models/                     class lists + pokeapi_slugs.json (weights not committed)
docs/                       Final_Report.pdf, this spec
```

Removed: `app.py`, `classifier.py`, `static/`, `test_pokemon_model_gui.py`, `Dockerfile`,
`.dockerignore`, `DEPLOY.md`, `requirements-app.txt`. README is rewritten for the new stack.

## Inference service (Flask)

**Models.** Both checkpoints are exported to ONNX by `training/export_onnx.py`:

- `convnext_gen1-3.onnx` — from `best_sketch.pth`, 386 classes (`best_sketch_classes.txt`), ImageNet normalization.
- `resnet18_gen1.onnx` — from `fine_tuned_with_pencil_sketch.pth`, `class_names.txt`, `[0.5,0.5,0.5]` normalization.

The ONNX files are published as GitHub Release assets (too large for git). The deploy workflow
downloads them into the Flask package. Locally, the developer downloads them into `models/`.
`export_onnx.py` also runs a one-time parity check: ONNX and PyTorch must agree on top-1 for a set
of sample images. This check is not part of CI.

**Preprocessing** reproduces the current torchvision pipeline exactly, using Pillow + NumPy:
convert to RGB → resize shorter side to 256 (bilinear) → center crop 224 → scale to [0,1] → normalize → NCHW float32.

**Routing** (ported unchanged from `app.py`):

1. Run ConvNeXt; take top-k.
2. If its top-1 is a Gen 1 name (first 151 entries of `class_names_v2.txt`), run ResNet-18.
3. Map legacy names through `LEGACY_FIX` (`Farfetchd`→`Farfetch'd`, `Mime`/`MrMime`→`Mr. Mime`, `Nidoran1`→`Nidoran-F`, `Nidoran2`→`Nidoran-M`).
4. If ResNet-18 top-1 confidence ≥ `LEGACY_TRUST` (env, default 0.55), return its results with `model: "legacy-gen1"`; otherwise ConvNeXt results with `model: "gen1-3"`.

Routing is a pure function over model outputs so it can be tested without real models.

**Endpoints**

- `POST /predict` — multipart field `file`, optional `top_k` (default 3). Requires header `X-Api-Key` matching env `INFERENCE_API_KEY`; otherwise 401.
  Response: `{"model": "gen1-3", "predictions": [{"name": "Pikachu", "confidence": 97.31}, ...]}`. Unreadable image → 400.
- `GET /health` — 200 once models are loaded (used by App Service health check). No key required.

Served by gunicorn on App Service.

## Web app (ASP.NET Core MVC)

**Styling.** Bootstrap 5 + jQuery from the MVC template, plus a small theme stylesheet (dark background, yellow accent, as today). Mobile-first layout.

**`/` — Guess page**

- Drag-drop, clipboard paste, file pick, and camera capture on mobile (`<input type="file" accept="image/*" capture="environment">`).
- Preview → jQuery `$.ajax` POST multipart to `/api/guess`.
- Shows top-3 confidence bars and a badge for which model answered. Top-1 ≥ 60%: fetch sprite + cry from PokéAPI (slug from `pokeapi_slugs.json`) and show/play them. Below 60%: "Are you sure this isn't a Digimon?" message with the best guess.
- ✓ Correct / ✗ Wrong buttons (hidden if `guessId` is null). Wrong reveals an `<input list>` + `<datalist>` of the 386 names; submit POSTs to `/api/guess/{id}/feedback`.

**`/Stats` — Razor-rendered**

- Total guesses, guesses with feedback, overall accuracy (correct / with feedback).
- Accuracy by model (`gen1-3` vs `legacy-gen1`) and by generation of the predicted Pokémon.
- Top 10 most-guessed Pokémon.
- Top 10 mix-ups: (predicted → actual) pairs among wrong guesses, by count.

Generation is derived from the dex-order list (`class_names_v2.txt`: indexes 0–150 Gen 1, 151–250 Gen 2, 251–385 Gen 3), and is not stored.

**API**

| Endpoint | Behavior |
|---|---|
| `POST /api/guess` | multipart `file`. Validates ≤ 5 MB and `image/*` content type (else 400). Forwards to Flask. Saves a `Guess` row from top-1. Returns `{guessId, model, predictions}`. Rate-limited per client IP (fixed window, 10/min; 429 when exceeded). |
| `POST /api/guess/{id}/feedback` | JSON `{isCorrect: bool, actualName?: string}`. 404 for unknown id. 409 if feedback was already set. 400 if `isCorrect` is false and `actualName` is missing or not one of the 386 names. On `isCorrect: true`, `ActualName` is set to `PredictedName`. |
| `GET /api/stats` | Same numbers as the Stats page. JSON by default; XML when `Accept: application/xml`. |

**Inference client.** A typed `HttpClient` registered with base URL `Inference:BaseUrl` and `X-Api-Key` from `Inference:ApiKey` (app settings / user-secrets, never committed). Timeout is 30 s.

## Data

EF Core, one table. SQLite locally, Azure SQL in production (provider chosen by connection string config). Migrations are applied at startup with `Database.Migrate()` (single instance).

**`Guesses`**

| Column | Type | Notes |
|---|---|---|
| Id | int PK identity | |
| CreatedAt | datetime2 | UTC |
| PredictedName | nvarchar(40) | top-1 |
| Confidence | decimal(5,2) | percent |
| Model | nvarchar(20) | `gen1-3` / `legacy-gen1` |
| IsCorrect | bit, null | null until feedback |
| ActualName | nvarchar(40), null | set with feedback |

## Error handling

| Failure | Behavior |
|---|---|
| Flask unreachable / timeout / 5xx | `/api/guess` returns 503; UI: "The Pokédex is waking up, try again in a few seconds." |
| Flask 400 (unreadable image) | `/api/guess` returns 400; UI: "Couldn't read that image." |
| DB save fails on guess | Log the error, still return predictions with `guessId: null` (feedback hidden). |
| PokéAPI fails | Show results without sprite/cry. |
| Rate limit hit | 429; UI: "Slow down, trainer, try again in a minute." |

## Testing

**Flask (pytest)**

- Routing: Gen-1 deferral above/below threshold, non-Gen-1 skips legacy, `LEGACY_FIX` mapping.
- Preprocessing: output shape `(1, 3, 224, 224)` and dtype float32.
- `/predict`: 401 without key, 400 on non-image bytes, 200 shape on a sample image (with stub models).

**.NET (xUnit + `WebApplicationFactory`, in-memory SQLite, fake `HttpMessageHandler` for inference)**

- Guess: happy path saves a row and returns `guessId`; Flask down → 503; Flask 400 → 400; oversize / non-image → 400.
- Feedback: correct path, wrong with valid name, wrong with invalid/missing name → 400, unknown id → 404, second feedback → 409.
- Stats: aggregates over seeded rows (accuracy, by model, by generation, top guessed, top mix-ups).
- `/api/stats` returns XML for `Accept: application/xml`.

## Azure & CI/CD

**Resources** (created by `infra/setup.sh`, all in one resource group so teardown is one command):

- One App Service plan, Linux, B1, Always On enabled.
- Web app `pokeguess-web` (.NET 10 runtime) and web app `pokeguess-inference` (Python 3.12), on that plan.
- Azure SQL logical server + database using the free serverless offer.
- App settings: connection string and `Inference__BaseUrl` / `Inference__ApiKey` on the web app; `INFERENCE_API_KEY` and `LEGACY_TRUST` on the inference app; health check path `/health` on the inference app.

Note: the Azure free trial's $200 credit expires after 30 days. After that, B1 runs about $13/month on pay-as-you-go. Delete the resource group to stop charges and re-run `setup.sh` to recreate.

**GitHub Actions**

- `ci.yml` (pull requests): `dotnet test`; `pytest` for `inference/` (with stub models, no ONNX download).
- `deploy.yml` (push to `main`): run tests, download ONNX models from the GitHub Release, deploy `inference/` and the published web app to their App Services. Azure login via OIDC federated credentials (`azure/login`), so no Azure secrets are stored in GitHub.

**Local dev:** `flask --app inference/app.py run` and `dotnet run --project src/PokeGuess.Web` with SQLite and user-secrets. No Docker.
