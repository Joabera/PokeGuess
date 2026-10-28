# PokeGuess .NET Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FastAPI/PyTorch serving stack with an ASP.NET Core MVC web app (feedback + stats in SQL) that calls a Flask + ONNX Runtime inference service, both deployed to Azure App Service by GitHub Actions.

**Architecture:** The browser talks only to the MVC app (jQuery AJAX → `/api/*`). MVC forwards images to Flask (`POST /predict`, `X-Api-Key`), stores each guess in a single `Guesses` table via EF Core, and renders a Stats page from aggregate queries. Flask runs both models (ConvNeXt Gen 1–3 + ResNet-18 Gen 1) as ONNX and keeps the existing auto-routing.

**Tech Stack:** .NET 10, ASP.NET Core MVC, EF Core 10 (SQLite / SQL Server), xUnit, Bootstrap 5 + jQuery (template defaults); Python 3.12, Flask, gunicorn, onnxruntime, Pillow, NumPy, pytest; PyTorch (export only); Azure App Service (Linux B1), Azure SQL free offer, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-28-pokeguess-dotnet-rebuild-design.md`

## Global Constraints

- .NET 10 (`net10.0`), Python 3.12 on App Service (local 3.12+ fine).
- No SSO, no user accounts, never store uploaded images.
- Canonical name list: `models/class_names_v2.txt` (386 names, dex order; idx 0–150 Gen 1, 151–250 Gen 2, 251–385 Gen 3).
- ConvNeXt class order: `models/best_sketch_classes.txt`; legacy class order: `models/class_names.txt` (154 lines).
- `LEGACY_FIX = {"Farfetchd": "Farfetch'd", "Mime": "Mr. Mime", "MrMime": "Mr. Mime", "Nidoran1": "Nidoran-F", "Nidoran2": "Nidoran-M"}`; `LEGACY_TRUST` env default `0.55` (compared against percent, i.e. 55.0).
- Normalization: ConvNeXt `mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)`; ResNet-18 `mean=std=(0.5,0.5,0.5)`.
- Confidence values are percentages rounded to 2 decimals.
- Uploads: ≤ 5 MB (5 × 1024 × 1024 bytes), content type starting `image/`.
- Rate limit `/api/guess`: fixed window, 10 per minute per client IP, 429.
- Low-confidence threshold in UI: top-1 < 60 → Digimon message.
- Secrets (`Inference:ApiKey`, `INFERENCE_API_KEY`, SQL password) never committed: user-secrets locally, App Service settings in Azure.
- User-facing copy (exact): `The Pokédex is waking up, try again in a few seconds.` / `Couldn't read that image.` / `Slow down, trainer, try again in a minute.` / `Are you sure this isn't a Digimon? 😕 (best guess: {name}, {confidence}%)`.
- ONNX files and `.pth` weights are never committed; ONNX files are published as GitHub Release `models-v1` assets.

## Review Focus

1. **Every user behind Azure's front end shares one IP** → the rate limiter must key on `X-Forwarded-For`, or one user's 10 guesses lock out everyone. Test in Task 6.
2. **Phone photos are big (3–12 MB) and often transparent PNG sketches** → client downscales images whose longest side > 1024 px before upload (PNG stays PNG); the server still rejects > 5 MB. Server test in Task 6; RGBA/palette/grayscale inputs tested in Task 3.
3. **Double-tap on ✓/✗ or two tabs** → second feedback must be 409 even when concurrent; use a conditional `UPDATE ... WHERE IsCorrect IS NULL` (never load-check-save). Sequential test in Task 7; reviewers check the implementation for the atomic update.
4. **Stats with zero feedback** → accuracy is `null` (shown as "—"), never a divide-by-zero 500. Test in Task 8.
5. **Feedback name typed in different case (`pikachu`)** → accepted and stored as canonical `Pikachu`. Test in Task 7.

---

## File Structure

```
.gitignore                                   modify: add *.onnx, *.db, bin/, obj/, inference/models/
README.md                                    rewrite (Task 10)
training/                                    moved from scripts/ + legacy/
training/export_onnx.py                      .pth → .onnx + parity check
inference/preprocess.py                      PIL image → NCHW float32 array
inference/routing.py                         pure two-model routing + LEGACY_FIX
inference/model.py                           OnnxClassifier, load_class_names
inference/app.py                             Flask create_app factory
inference/requirements.txt / requirements-dev.txt / pytest.ini
inference/tests/test_preprocess.py, test_routing.py, test_app.py
PokeGuess.sln
src/PokeGuess.Web/Program.cs                 DI, DB init, rate limiter, forwarded headers
src/PokeGuess.Web/Data/Guess.cs, PokeGuessDb.cs, Migrations/
src/PokeGuess.Web/Services/Pokedex.cs        name list, canonical lookup, generation
src/PokeGuess.Web/Services/InferenceClient.cs
src/PokeGuess.Web/Services/StatsService.cs   + Stats DTO classes
src/PokeGuess.Web/Controllers/GuessApiController.cs, StatsController.cs, HomeController.cs (template)
src/PokeGuess.Web/Views/Home/Index.cshtml, Views/Stats/Index.cshtml
src/PokeGuess.Web/wwwroot/js/guess.js, wwwroot/css/site.css
src/PokeGuess.Web.Tests/PokeGuessFactory.cs, FakeInferenceHandler.cs, *Tests.cs
infra/setup.sh
.github/workflows/ci.yml, deploy.yml
```

---

### Task 1: Restructure the repo

**Files:**
- Move: `scripts/*` → `training/`, `legacy/*` → `training/legacy/`
- Delete: `app.py`, `classifier.py`, `static/`, `test_pokemon_model_gui.py`, `Dockerfile`, `.dockerignore`, `DEPLOY.md`, `requirements-app.txt`
- Modify: `.gitignore`, `training/scrape_dataset.py:60` (comment path), `README.md` (paths only; full rewrite in Task 10)

- [ ] **Step 1:** `git mv scripts training` then `git mv legacy training/legacy`; `git rm` the files listed above. Move `requirements.txt` → `training/requirements.txt`.
- [ ] **Step 2:** Update the "run as `python scripts/<name>.py`" comment in `training/scrape_dataset.py` and any `scripts/` mentions in `training/*.py` docstrings to `training/`. `train.py`'s `from sketch import RandomizedSketch` keeps working (same directory).
- [ ] **Step 3:** Append to `.gitignore`: `*.onnx`, `*.db`, `bin/`, `obj/`, `inference/models/`, `publish/`.
- [ ] **Step 4: Verify.** Run `git grep -n "scripts/\|classifier\|app\.py"` → expected: hits only in `docs/` and README. Run `python -m py_compile training/*.py` → no output.
- [ ] **Step 5: Commit** `chore: move training code to training/, remove FastAPI serving stack`.

---

### Task 2: Export both models to ONNX with a parity check

**Files:**
- Create: `inference/preprocess.py` (the function only; its tests come in Task 3), `training/export_onnx.py`

**Interfaces:**
- Produces: `models/convnext_gen1-3.onnx`, `models/resnet18_gen1.onnx` (input `"input"` float32 `[1,3,224,224]`, output `"logits"` `[1,N]`); GitHub Release `models-v1` with both files.
- Produces: `preprocess.to_tensor(image: PIL.Image.Image, mean: tuple[float,float,float], std: tuple[float,float,float]) -> np.ndarray` shape `(1,3,224,224)` float32; constants `IMAGENET_NORM`, `HALF_NORM` as `(mean, std)` tuples.

- [ ] **Step 1:** Create a training venv and copy the weights: `python -m venv .venv-train`, then `.venv-train\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` and `pip install onnx onnxruntime pillow numpy`. Copy `C:\Users\Joab\Documents\PokeGuess\PokeGuess\models\best_sketch.pth` into `models/` (gitignored).
- [ ] **Step 2: Implement `to_tensor`** in `inference/preprocess.py`, matching torchvision `Resize(256) → CenterCrop(224) → ToTensor → Normalize` exactly. Algorithm (do not substitute):
  ```python
  img = image.convert("RGB")            # also handles RGBA, P, L, and the first frame of animated GIFs
  w, h = img.size
  if w <= h: nw, nh = 256, int(256 * h / w)
  else:      nw, nh = int(256 * w / h), 256
  img = img.resize((nw, nh), Image.BILINEAR)
  left, top = int(round((nw - 224) / 2.0)), int(round((nh - 224) / 2.0))
  img = img.crop((left, top, left + 224, top + 224))
  arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std   # HWC
  return arr.transpose(2, 0, 1)[None].astype(np.float32)
  ```
- [ ] **Step 3: Implement `training/export_onnx.py`.** Build each backbone with the same `_build_backbone` as the old `classifier.py` (`git show bc2f78c:classifier.py`; ConvNeXt: replace `classifier[2]`; ResNet: replace `fc`). Load the state dict on CPU, `eval()`, and call `torch.onnx.export(model, torch.zeros(1,3,224,224), out, input_names=["input"], output_names=["logits"], opset_version=17)`. CLI: no arguments; exports both pairs `(models/best_sketch.pth, models/best_sketch_classes.txt, IMAGENET_NORM) → models/convnext_gen1-3.onnx` and `(models/fine_tuned_with_pencil_sketch.pth, models/class_names.txt, HALF_NORM) → models/resnet18_gen1.onnx`.
- [ ] **Step 4: Add the parity check** to the same script, run after each export and raising `SystemExit(1)` on failure:
  - Preprocessing parity: for 3 synthetic images (an RGB gradient 640×480, an RGBA 300×500 with transparency, a grayscale 1000×1000), `np.allclose(to_tensor(img, *norm), torchvision_transform(img).unsqueeze(0).numpy(), atol=1e-4)`.
  - Model parity: for those 3 tensors, `np.allclose(onnx_logits, torch_logits, atol=1e-3)` and `argmax` equal.
  - Prints `PARITY OK <model>` per model.
  
  Import `preprocess` with `sys.path.insert(0, "inference")`. Run from the repo root.
- [ ] **Step 5: Run it.** `.venv-train\Scripts\python training/export_onnx.py` → expected: `PARITY OK convnext_gen1-3`, `PARITY OK resnet18_gen1`, and both `.onnx` files in `models/` (~110 MB and ~45 MB).
- [ ] **Step 6: Commit** `feat: ONNX export with preprocessing/model parity check` (script and preprocess.py only; the .onnx files are gitignored).
- [ ] **Step 7: Publish the release. Ask the user before running** (it's outward-facing): `gh release create models-v1 models/convnext_gen1-3.onnx models/resnet18_gen1.onnx --title "Model weights v1" --notes "ONNX exports of best_sketch.pth and fine_tuned_with_pencil_sketch.pth"`.

---

### Task 3: Flask preprocessing tests + routing

**Files:**
- Create: `inference/routing.py`, `inference/requirements.txt`, `inference/requirements-dev.txt`, `inference/pytest.ini`, `inference/tests/test_preprocess.py`, `inference/tests/test_routing.py`

**Interfaces:**
- Consumes: `preprocess.to_tensor`, `IMAGENET_NORM` (Task 2).
- Produces: `Prediction = tuple[str, float]`; `LEGACY_FIX: dict[str, str]`; `route(new_preds: list[Prediction], legacy_predict: Callable[[], list[Prediction]], gen1_names: set[str], legacy_trust_pct: float) -> tuple[list[Prediction], str]` returning `(preds, "legacy-gen1" | "gen1-3")`. `legacy_predict` is called only when `new_preds[0][0] in gen1_names`.

- [ ] **Step 1:** `requirements.txt`: `flask>=3.0`, `gunicorn>=22`, `onnxruntime>=1.18`, `pillow>=10`, `numpy>=1.26`. `requirements-dev.txt`: `-r requirements.txt`, `pytest>=8`. `pytest.ini`: `[pytest]` / `pythonpath = .` / `testpaths = tests`. Create `inference/.venv` and install the dev requirements.
- [ ] **Step 2: Write the failing tests.**
  ```python
  # test_preprocess.py
  @pytest.mark.parametrize("mode,size", [("RGB",(640,480)),("RGBA",(300,500)),("P",(100,100)),("L",(1000,1000))])
  def test_to_tensor_shape_and_dtype(mode, size):
      out = to_tensor(Image.new(mode, size), *IMAGENET_NORM)
      assert out.shape == (1, 3, 224, 224) and out.dtype == np.float32

  # test_routing.py   GEN1 = {"Pikachu", "Mr. Mime", "Nidoran-F"}
  def test_non_gen1_never_calls_legacy():  # new top "Treecko"; legacy_predict raises if called → result model "gen1-3"
  def test_gen1_legacy_confident_wins():   # new [("Pikachu",70.0)], legacy [("Pikachu",55.0)], trust 55.0 → ("legacy-gen1", legacy preds)
  def test_gen1_legacy_below_trust_uses_new():  # legacy top 54.99 → "gen1-3", new preds unchanged
  def test_legacy_names_are_fixed():       # legacy [("MrMime",90.0),("Nidoran1",5.0)] → names ["Mr. Mime","Nidoran-F"]
  ```
- [ ] **Step 3: Run** `pytest` in `inference/` → expected: routing tests FAIL (ImportError on `routing`); preprocess tests PASS.
- [ ] **Step 4: Implement `route`** in `inference/routing.py` per Global Constraints (apply `LEGACY_FIX` to every legacy prediction name; compare `>=`).
- [ ] **Step 5: Run** `pytest` → all PASS.
- [ ] **Step 6: Commit** `feat(inference): two-model routing and preprocessing tests`.

---

### Task 4: Flask app + ONNX classifier

**Files:**
- Create: `inference/model.py`, `inference/app.py`, `inference/tests/test_app.py`

**Interfaces:**
- Consumes: `to_tensor`, `IMAGENET_NORM`, `HALF_NORM`, `route`.
- Produces: `load_class_names(path: str) -> list[str]`; `class OnnxClassifier(model_path: str, class_names: list[str], norm: tuple)` with `predict(image: PIL.Image.Image, top_k: int) -> list[Prediction]` (softmax over logits, top-k by probability, confidence in percent). `create_app(new_clf=None, legacy_clf=None, gen1_names=None, api_key=None, legacy_trust_pct=None) -> Flask`. Any `None` is loaded from env: `MODEL_DIR` (default `models`), `INFERENCE_API_KEY` (required; raise at startup if missing), `LEGACY_TRUST` (default `0.55`, ×100). Files under `MODEL_DIR`: `convnext_gen1-3.onnx`, `best_sketch_classes.txt`, `resnet18_gen1.onnx`, `class_names.txt`, `class_names_v2.txt` (first 151 = gen1).
- HTTP contract (consumed by Task 6): `POST /predict` multipart `file`, query `top_k` (default 3, clamp 1–10), header `X-Api-Key`. 200 → `{"model": str, "predictions": [{"name": str, "confidence": float}]}`, confidence rounded to 2. 401 `{"error":"unauthorized"}`; 400 `{"error":"Could not read image file."}`. `GET /health` → 200 `{"status":"ok"}`, no key.

- [ ] **Step 1: Write the failing tests** with a stub classifier (`predict` returns a fixed list) and `api_key="test-key"`:
  ```python
  def test_health_no_key(client): assert client.get("/health").status_code == 200
  def test_predict_requires_key(client): # no header → 401; wrong key → 401
  def test_predict_rejects_non_image(client): # b"not an image" → 400, json["error"] == "Could not read image file."
  def test_predict_ok(client):  # 64x64 PNG from Pillow → 200; json["model"] == "gen1-3"; len(predictions) == 3; confidences rounded to 2
  def test_predict_routes_to_legacy(client_gen1):  # stub new top "Pikachu", stub legacy 80.0 → model "legacy-gen1"
  ```
- [ ] **Step 2: Run** `pytest tests/test_app.py` → FAIL (ImportError).
- [ ] **Step 3: Implement `model.py`** (`onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])`; numerically stable softmax) and **`app.py`** (key check with `hmac.compare_digest`; open the image with `Image.open(io.BytesIO(...))` + `.load()`, and catch `UnidentifiedImageError, OSError` → 400).
- [ ] **Step 4: Run** `pytest` → all PASS.
- [ ] **Step 5: Smoke test with real models.** Copy the five model files into `inference/models/`, then `$env:INFERENCE_API_KEY="dev"; flask --app inference/app.py run` and `curl -H "X-Api-Key: dev" -F file=@<any pokemon png> http://127.0.0.1:5000/predict` → expected: 200 with a sensible top-1.
- [ ] **Step 6: Commit** `feat(inference): Flask /predict and /health on ONNX Runtime`.

---

### Task 5: .NET solution, data model, Pokedex, test harness

**Files:**
- Create: `PokeGuess.sln`, `src/PokeGuess.Web/` (via `dotnet new mvc`), `src/PokeGuess.Web.Tests/` (via `dotnet new xunit`), `Data/Guess.cs`, `Data/PokeGuessDb.cs`, `Services/Pokedex.cs`, `Data/Migrations/*`, test `PokeGuessFactory.cs`, `FakeInferenceHandler.cs`, `PokedexTests.cs`
- Modify: `Program.cs`, `appsettings.json`, `appsettings.Development.json`, `PokeGuess.Web.csproj`

**Interfaces:**
- Produces: `Guess { int Id; DateTime CreatedAt; string PredictedName; decimal Confidence; string Model; bool? IsCorrect; string? ActualName }` (max lengths 40/20/40, `HasPrecision(5,2)`). `PokeGuessDb : DbContext` with `DbSet<Guess> Guesses`.
- Produces: `sealed class Pokedex(IReadOnlyList<string> dexOrderNames)` with `static Pokedex Load(string path)`, `IReadOnlyList<string> Names`, `bool TryCanonical(string name, out string canonical)` (trimmed, case-insensitive), `int GenerationOf(string name)` (1/2/3, 0 if unknown). Registered as a singleton from config `Pokedex:Path` (default `Data/class_names_v2.txt`).
- Produces: config keys `Database:Provider` (`Sqlite` | `SqlServer`), `ConnectionStrings:Default`, `Inference:BaseUrl`, `Inference:ApiKey`, `RateLimit:GuessPerMinute` (10). `public partial class Program {}` at the end of `Program.cs`.
- Produces (tests): `PokeGuessFactory : WebApplicationFactory<Program>`, **one new instance per test** (created in the test class constructor, disposed in `Dispose`) so DB and rate-limiter state never leak between tests. It replaces `PokeGuessDb` options with one shared open in-memory `SqliteConnection("DataSource=:memory:")`, exposes `FakeInferenceHandler Inference` and `PokeGuessDb CreateDb()`. `FakeInferenceHandler : HttpMessageHandler` with a settable `Func<HttpRequestMessage, HttpResponseMessage> Respond` (default: 200 `{"model":"gen1-3","predictions":[{"name":"Pikachu","confidence":97.31},{"name":"Raichu","confidence":1.5},{"name":"Pichu","confidence":0.4}]}`) and `HttpRequestMessage? LastRequest`.

- [ ] **Step 1: Scaffold.** `dotnet new sln -n PokeGuess`; `dotnet new mvc -o src/PokeGuess.Web -f net10.0`; `dotnet new xunit -o src/PokeGuess.Web.Tests -f net10.0`; add both to the sln and reference Web from Tests. Packages: Web gets `Microsoft.EntityFrameworkCore.Sqlite`, `.SqlServer`, `.Design`; Tests get `Microsoft.AspNetCore.Mvc.Testing`, `Microsoft.EntityFrameworkCore.Sqlite`. Use 10.0.x versions throughout.
- [ ] **Step 2:** In the csproj, link `..\..\models\class_names_v2.txt` → `Data\class_names_v2.txt` and `..\..\models\pokeapi_slugs.json` → `wwwroot\data\pokeapi_slugs.json` as `Content` with `CopyToOutputDirectory=PreserveNewest` (so they publish).
- [ ] **Step 3: Write the failing tests** `PokedexTests`:
  ```csharp
  [Fact] Load_reads_386_names()            // Pokedex.Load(path).Names.Count == 386, [0]=="Bulbasaur"
  [Theory] [InlineData("Mew",1)] [InlineData("Chikorita",2)] [InlineData("Celebi",2)] [InlineData("Treecko",3)] [InlineData("Deoxys",3)] [InlineData("Agumon",0)]
  GenerationOf_uses_dex_order(string name, int gen)
  [Fact] TryCanonical_is_case_insensitive() // " pikachu " → true, "Pikachu"; "mr. mime" → "Mr. Mime"; "Agumon" → false
  [Fact] App_starts_and_serves_home()      // factory client GET "/" → 200
  ```
- [ ] **Step 4: Run** `dotnet test` → FAIL (compile errors).
- [ ] **Step 5: Implement** `Guess`, `PokeGuessDb`, `Pokedex`, and in `Program.cs`: register the DbContext by `Database:Provider` (`UseSqlite` / `UseSqlServer` with `ConnectionStrings:Default`). On startup: SqlServer → `Database.Migrate()`; Sqlite → `Database.EnsureCreated()`. Rationale: migrations are provider-specific, so they are generated for SQL Server only, and SQLite (dev + tests) uses EnsureCreated. `appsettings.json`: `Database:Provider=SqlServer`, `RateLimit:GuessPerMinute=10`, `Inference:BaseUrl=""`. `appsettings.Development.json`: `Database:Provider=Sqlite`, `ConnectionStrings:Default="Data Source=pokeguess.db"`, `Inference:BaseUrl="http://127.0.0.1:5000"`. Run `dotnet user-secrets set Inference:ApiKey dev` for the Web project.
- [ ] **Step 6: Generate the migration.** `dotnet tool install --global dotnet-ef`, then from `src/PokeGuess.Web`: `dotnet ef migrations add Initial -o Data/Migrations -- --Database:Provider=SqlServer "--ConnectionStrings:Default=Server=.;Database=design;Trusted_Connection=True"`. Expected: a migration that creates `Guesses` with `decimal(5,2)` and `nvarchar(40)` columns.
- [ ] **Step 7: Run** `dotnet test` → all PASS.
- [ ] **Step 8: Commit** `feat(web): MVC scaffold, Guess model, Pokedex, test harness`.

---

### Task 6: Inference client + `POST /api/guess`

**Files:**
- Create: `Services/InferenceClient.cs`, `Controllers/GuessApiController.cs`, test `GuessApiTests.cs`
- Modify: `Program.cs` (typed HttpClient, rate limiter, forwarded headers)

**Interfaces:**
- Consumes: `PokeGuessDb`, `Guess`, `FakeInferenceHandler`, `PokeGuessFactory`, Flask HTTP contract (Task 4).
- Produces: `record Prediction(string Name, decimal Confidence)`; `record InferenceResult(string Model, IReadOnlyList<Prediction> Predictions)`; `InferenceClient.PredictAsync(Stream image, string fileName, string contentType, CancellationToken ct) -> Task<InferenceResult>`, which throws `InferenceBadImageException` on Flask 400 and `InferenceUnavailableException` on timeout, connection failure, or any other non-2xx. Typed HttpClient: `BaseAddress = Inference:BaseUrl`, default header `X-Api-Key = Inference:ApiKey`, `Timeout = 30s`.
- Produces: `POST /api/guess` (multipart `file`) → 200 `{guessId: int|null, model: string, predictions: [{name, confidence}]}` (camelCase JSON). Errors: `{ "error": "<exact copy>" }` with 400 / 503 / 429. Rate-limiter policy name `"guess"`.

- [ ] **Step 1: Write the failing tests**:
  ```csharp
  Guess_happy_path_saves_row()        // 200; guessId > 0; predictions[0].name=="Pikachu"; DB row PredictedName "Pikachu", Confidence 97.31m, Model "gen1-3", IsCorrect null
  Guess_forwards_api_key()            // Inference.LastRequest.Headers["X-Api-Key"] == configured key
  Guess_rejects_oversize()            // 5*1024*1024 + 1 bytes, image/png → 400; Inference.LastRequest is null
  Guess_rejects_non_image_type()      // text/plain → 400 "Couldn't read that image."
  Guess_inference_400_maps_to_400()   // Respond → 400 → 400 "Couldn't read that image."
  Guess_inference_down_maps_to_503()  // Respond throws HttpRequestException → 503 "The Pokédex is waking up, try again in a few seconds."
  Guess_db_failure_still_returns_predictions() // run "DROP TABLE Guesses" on the factory connection → 200, guessId null
  Guess_rate_limited_per_forwarded_ip() // 10 requests with X-Forwarded-For 203.0.113.1 → 200; 11th → 429 "Slow down, trainer, try again in a minute."; 1st request with X-Forwarded-For 203.0.113.2 → 200
  ```
- [ ] **Step 2: Run** `dotnet test --filter GuessApiTests` → FAIL.
- [ ] **Step 3: Implement.** In `Program.cs`: `UseForwardedHeaders` (XForwardedFor, with `KnownIPNetworks` and `KnownProxies` cleared, since App Service's front end is the only proxy) before `UseRateLimiter`. Fixed-window limiter `"guess"` partitioned on `RemoteIpAddress?.ToString() ?? "unknown"`, `PermitLimit = RateLimit:GuessPerMinute`, 1-minute window, `QueueLimit = 0`, and an `OnRejected` that writes 429 + the JSON error copy. Controller: `[RequestSizeLimit(6 * 1024 * 1024)]` + `[EnableRateLimiting("guess")]`. Validate size and content type first, call the client, map exceptions to codes, then save the row inside `try { … } catch (DbUpdateException) { log; guessId = null }` (also catch `SqliteException`/`SqlException` via the base `DbException`).
- [ ] **Step 4: Run** `dotnet test` → all PASS.
- [ ] **Step 5: Commit** `feat(web): POST /api/guess with validation, rate limiting, error mapping`.

---

### Task 7: `POST /api/guess/{id}/feedback`

**Files:**
- Modify: `Controllers/GuessApiController.cs`
- Create: test `FeedbackApiTests.cs`

**Interfaces:**
- Consumes: `Pokedex.TryCanonical`, `PokeGuessDb`.
- Produces: `POST /api/guess/{id:int}/feedback`, JSON `record FeedbackRequest(bool IsCorrect, string? ActualName)` → 204. Errors: 400 `{error:"Pick a Pokémon from the list."}`, 404, 409 `{error:"Feedback already recorded."}`.

- [ ] **Step 1: Write the failing tests** (seed one `Guess` with PredictedName "Pikachu" for each):
  ```csharp
  Correct_sets_actual_to_predicted()     // {isCorrect:true} → 204; row IsCorrect true, ActualName "Pikachu"
  Wrong_with_valid_name_canonicalizes()  // {isCorrect:false, actualName:"raichu"} → 204; ActualName "Raichu"
  Wrong_with_missing_or_unknown_name_is_400() // null, "", "Agumon" → 400; row unchanged
  Unknown_id_is_404()
  Second_feedback_is_409()               // two sequential posts → 204 then 409
  ```
  (No concurrent test: the shared in-memory SQLite connection isn't thread-safe. The concurrency guarantee comes from the conditional update in Step 3.)
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Validate the name first, then write with `db.Guesses.Where(g => g.Id == id && g.IsCorrect == null).ExecuteUpdateAsync(...)`. If 0 rows are affected, `AnyAsync(g => g.Id == id)` decides 409 vs 404. This is the atomic one-time guarantee; don't load, check, then save.
- [ ] **Step 4: Run** `dotnet test` → all PASS.
- [ ] **Step 5: Commit** `feat(web): one-time guess feedback`.

---

### Task 8: Stats service, `/Stats` page, `/api/stats` JSON/XML

**Files:**
- Create: `Services/StatsService.cs`, `Controllers/StatsController.cs`, `Views/Stats/Index.cshtml`, test `StatsTests.cs`
- Modify: `Program.cs` (`AddControllersWithViews().AddXmlSerializerFormatters()`, register `StatsService` scoped), `Views/Shared/_Layout.cshtml` (nav: Guess, Stats)

**Interfaces:**
- Consumes: `PokeGuessDb`, `Pokedex.GenerationOf`.
- Produces: DTOs as **classes with public get/set and `List<T>`** (XmlSerializer cannot handle records): `StatsDto { int TotalGuesses; int WithFeedback; double? Accuracy; List<GroupAccuracy> ByModel; List<GroupAccuracy> ByGeneration; List<NameCount> MostGuessed; List<MixUp> TopMixUps }`, `GroupAccuracy { string Key; int WithFeedback; int Correct; double? Accuracy }`, `NameCount { string Name; int Count }`, `MixUp { string Predicted; string Actual; int Count }`. Accuracy = percent rounded to 1 decimal, `null` when WithFeedback == 0. `StatsService.GetAsync(CancellationToken) -> Task<StatsDto>`.
- Rules: `ByModel` lists only models present, ordered by Key. `ByGeneration` always has `"Gen 1"`, `"Gen 2"`, `"Gen 3"` in that order: group by PredictedName in SQL, fold by generation in memory. `MostGuessed` is top 10 by count desc, then name asc. `TopMixUps` is top 10 over rows with `IsCorrect == false`, grouped by (PredictedName, ActualName), count desc.
- Routes: `GET /Stats` (view), `GET /api/stats` (`Ok(dto)`, JSON default, XML for `Accept: application/xml`).

- [ ] **Step 1: Write the failing tests**:
  ```csharp
  Empty_db_has_null_accuracy()     // Total 0, Accuracy null, ByGeneration 3 entries all null accuracy; GET /api/stats → 200
  Aggregates_seeded_rows()         // seed: Pikachu legacy-gen1 correct ×3; Pikachu gen1-3 wrong→Raichu ×1; Treecko gen1-3 correct ×1; Mudkip gen1-3 no feedback ×1
                                   // Total 6, WithFeedback 5, Accuracy 80.0; ByModel gen1-3 {2 fb, 1 correct, 50.0}, legacy-gen1 {3,3,100.0}
                                   // ByGeneration Gen 1 {4,3,75.0}, Gen 2 {0,0,null}, Gen 3 {1,1,100.0}; MostGuessed[0] Pikachu 4; TopMixUps == [Pikachu→Raichu 1]
  Api_stats_returns_xml_when_asked() // Accept: application/xml → content-type application/xml, body contains "<TotalGuesses>6</TotalGuesses>"
  Stats_page_renders()             // GET /Stats → 200, contains "80.0%"; empty DB → contains "—"
  ```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the service, controller and view (Bootstrap cards for the headline numbers, `table-sm` tables for the breakdowns, `—` for null accuracy; tables wrapped in `.table-responsive` for phones).
- [ ] **Step 4: Run** `dotnet test` → all PASS.
- [ ] **Step 5: Commit** `feat(web): stats page and JSON/XML stats API`.

---

### Task 9: Guess page UI (jQuery, responsive)

**Files:**
- Modify: `Views/Home/Index.cshtml`, `wwwroot/css/site.css`, `Views/Shared/_Layout.cshtml` (`<script src="~/js/guess.js">` on Home only via `@section Scripts`)
- Create: `wwwroot/js/guess.js`
- Delete: template `Views/Home/Privacy.cshtml` + its action and nav link

**Interfaces:**
- Consumes: `POST /api/guess`, `POST /api/guess/{id}/feedback`, `/data/pokeapi_slugs.json`, `Pokedex.Names` (`@inject` into the view for the `<datalist id="pokemon-names">`), PokéAPI `https://pokeapi.co/api/v2/pokemon/{slug}` (sprite: `sprites.other["official-artwork"].front_default` fallback `sprites.front_default`; cry: `cries.latest` fallback `cries.legacy`).

Behavior to build, in that one file, using jQuery (`$.ajax` with `FormData`, `processData:false, contentType:false`):
- Drop zone (drag-over highlight), click-to-pick, `paste` on document (first `image/*` clipboard item), and `<input type="file" accept="image/*" capture="environment">` behind a "Use camera" button shown only on touch devices (`matchMedia("(pointer: coarse)")`).
- Before upload: if the longest side > 1024 px, downscale on a canvas to 1024 and re-encode (`image/png` stays PNG, otherwise JPEG at 0.9). Preview uses an object URL.
- Results: top-3 bars (width = confidence %), model badge (`gen1-3` → "Gen 1–3 model", `legacy-gen1` → "Gen 1 specialist"). Top-1 ≥ 60 → fetch sprite + cry by slug and show them. The cry button shows only if `new Audio().canPlayType('audio/ogg')` is non-empty (Safari). Top-1 < 60 → the exact Digimon copy with `confidence.toFixed(1)`.
- Feedback (only if `guessId` is non-null): ✓ → POST `{isCorrect:true}`; ✗ → reveal `<input list="pokemon-names">` + Submit → POST `{isCorrect:false, actualName}`. After 204: "Thanks! Logged." and disable the buttons. On 400, show the error text inline. On 409, treat as already logged.
- Error responses show `responseJSON.error` in the status line. Disable the drop zone while a request is in flight.
- `site.css`: CSS custom properties `--bg #0f172a`, `--panel #1e293b`, `--accent #fbbf24`, `--text #e2e8f0`, `--muted #94a3b8`, `--bar #38bdf8` applied over Bootstrap. Content is max-width 480px centered on desktop and full-width with a 16px gutter on phones. Touch targets ≥ 44px tall.

- [ ] **Step 1: Implement** the view, `guess.js` and `site.css`.
- [ ] **Step 2: Run** `dotnet test` → still all PASS (`App_starts_and_serves_home`).
- [ ] **Step 3: Manual check with both services running** (Flask from Task 4 Step 5 + `dotnet run --project src/PokeGuess.Web`), in the browser pane at desktop and the mobile preset (375×812): drop an image, paste an image, a large photo (> 1024 px) uploads downscaled (check the request size in the network panel), sprite + cry appear for a confident guess, a random non-Pokémon photo shows the Digimon line, ✓ and ✗ + datalist both save and the Stats page reflects them, a stopped Flask shows the 503 copy, there's no horizontal scroll on mobile, and the console has no errors.
- [ ] **Step 4: Commit** `feat(web): responsive jQuery guess page with feedback`.

---

### Task 10: Azure infra, CI/CD, README

**Files:**
- Create: `infra/setup.sh`, `.github/workflows/ci.yml`, `.github/workflows/deploy.yml`
- Rewrite: `README.md`

**Interfaces:**
- Consumes: Release `models-v1` (Task 2), Flask factory `create_app` (Task 4), config keys (Task 5).
- Produces: GitHub repo variables `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_WEBAPP_WEB`, `AZURE_WEBAPP_INFERENCE` (variables, not secrets, since none of them are secret).

- [ ] **Step 1: `infra/setup.sh`** (bash, `set -euo pipefail`; inputs `RG` default `pokeguess-rg`, `LOCATION` default `eastus`, `SUFFIX` default random 5 chars) does, in order:
  1. `az group create`; `az appservice plan create --is-linux --sku B1`.
  2. `az webapp create` ×2: `pokeguess-web-$SUFFIX` runtime `DOTNETCORE:10.0`, and `pokeguess-inference-$SUFFIX` runtime `PYTHON:3.12`; `az webapp config set --always-on true` on both.
  3. Inference: startup command `gunicorn --bind=0.0.0.0 --timeout 120 --workers 1 'app:create_app()'`, `--generic-configurations '{"healthCheckPath":"/health"}'`, app settings `INFERENCE_API_KEY=<openssl rand -hex 32>`, `LEGACY_TRUST=0.55`, `SCM_DO_BUILD_DURING_DEPLOYMENT=true`.
  4. `az sql server create` (admin password = `openssl rand -base64 24` + `Aa1!` suffix for complexity), firewall rule `AllowAzureServices` 0.0.0.0–0.0.0.0, `az sql db create --use-free-limit --free-limit-exhaustion-behavior AutoPause`.
  5. Web app settings: `Database__Provider=SqlServer`, connection string `Default` (type SQLAzure), `Inference__BaseUrl=https://<inference host>`, `Inference__ApiKey=<same key>`.
  6. OIDC: `az ad app create`, a service principal, a federated credential for `repo:Joabera/PokeGuess:ref:refs/heads/main`, and the `Contributor` role scoped to the RG; then `gh variable set` for the five variables.
  7. Prints both URLs. Generated passwords and keys are never echoed.
  
  Header comment: how to tear down (`az group delete -n $RG`), and the credit note from the spec.
- [ ] **Step 2: `ci.yml`** on `pull_request`: ubuntu-latest; `actions/setup-dotnet` `10.0.x` → `dotnet test`; `actions/setup-python` `3.12` → `pip install -r inference/requirements-dev.txt` → `pytest` in `inference/`.
- [ ] **Step 3: `deploy.yml`** on push to `main` (+ `workflow_dispatch`), `permissions: id-token: write, contents: read`: run the same tests; `dotnet publish src/PokeGuess.Web -c Release -o publish`; `gh release download models-v1 -p "*.onnx" -D inference/models` (env `GH_TOKEN: ${{ github.token }}`) + copy `models/best_sketch_classes.txt`, `class_names.txt`, `class_names_v2.txt` into `inference/models/`; `azure/login@v2` with the three ID variables; `azure/webapps-deploy@v3` for `publish/` → web and `inference/` → inference (exclude `tests/` and `.venv`).
- [ ] **Step 4: Verify the workflows locally:** run `actionlint` if it's installed (`go install` or the release binary); otherwise check them by eye against the steps above. `dotnet test` and `pytest` pass.
- [ ] **Step 5: README rewrite:** the live link placeholder (filled in after the first deploy), the architecture diagram from the spec, the requirement → code table, local dev (`flask --app inference/app.py run` + `dotnet run --project src/PokeGuess.Web`, user-secrets, downloading models with `gh release download models-v1`), training + export, deploy (`infra/setup.sh` then push to main), and teardown. Keep the Results table from the old README.
- [ ] **Step 6: Commit** `feat: Azure infra script, CI and deploy workflows, README`.
- [ ] **Step 7: Deploy. User-run (outward-facing, costs credits).** The user runs `az login` and then `bash infra/setup.sh`; we open a PR `feat/dotnet-rebuild` → `main` (CI runs), merge, and confirm the deploy succeeded, that the web URL guesses an image end to end, and that `/Stats` reflects it.
