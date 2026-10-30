# PokeGuess Live Mode — Design

**Date:** 2026-10-29
**Branch:** `feat/live-mode`
**Status:** Approved in brainstorming, pending spec review
**Builds on:** `2026-10-28-pokeguess-dotnet-rebuild-design.md` (the live site as deployed)

## Goal

Add a **Live** page where the user points a webcam or phone camera at a Pokémon picture, drawing, or card and sees the Pokémon recognized in real time, with green boxes and labels drawn over the video. The existing upload flow on the Guess page stays exactly as it is; Live is a second option.

**Success criteria**

- On a WebGPU browser, live recognition runs in real time: ~40 fps in guide-box mode, ≥ 4 fps in search mode with box refinement (spike: 9.2 fps searching, Eevee card recognized at 92% on a real webcam).
- Without WebGPU, live mode still works in guide-box mode via the CPU fallback (~2 fps).
- A **Snap** sends the current detection through the existing upload pipeline, so snapped guesses get the full two-model answer, ✓/✗ feedback, and appear in Stats tagged as `live`.
- The upload page's behavior and the existing tests are unchanged.

**Out of scope**

- Recording video, snapping several boxes at once.
- Tight, non-square outlines (would need a trained object detector).
- Running the Gen-1 ResNet-18 specialist in the browser.
- fp16 model variant (possible later optimization if the 113 MB download proves a problem).
- Logging live frames automatically (only Snaps are logged).

## Spike evidence (2026-10-29)

| Option | Measurement |
|---|---|
| Server `/api/guess` per frame | ~0.7 s warm round trip; one inference worker → ~1.5 guesses/s total across all users. Rejected. |
| Browser WebGPU, ConvNeXt fp32 | 25 ms per crop; 99 ms for a batch of 10 crops |
| Browser WASM (1 thread), ConvNeXt int8 | 463 ms per crop |
| Browser WASM (1 thread), ConvNeXt fp32 | 620 ms per crop |
| int8 vs fp32 accuracy | identical top-1 on 80 images (40 clean, 40 sketch-style); 28.5 MB vs 112.6 MB |
| int8 on WebGPU | ~680 ms (int8 ops fall back to CPU) — so WebGPU uses fp32 |
| Real webcam, search mode, WebGPU | 9.2 fps, 74 ms inference, Eevee card at 92% |

## Architecture

```
/Live page (browser)
  camera ──► detector (ONNX Runtime Web: WebGPU → WASM fallback)
               ConvNeXt live labels + boxes, nothing sent anywhere
  [Snap] ──► crop of the box ──► POST /api/guess (existing, + source=live)
                                   └► Flask: both models + routing ──► Guesses row (Source = live)
         ◄── server answer + ✓/✗ (existing feedback endpoint)
```

### Browser components

- `Views/Live/Index.cshtml` (+ `LiveController`): camera view, mode toggle, Snap/Resume, result panel. Nav becomes **Guess · Live · Stats**.
- `wwwroot/js/live/detector.js` — ES module of pure functions (no DOM, no camera), unit-tested with `node --test`:
  - `gridRegions(W, H)` → 1 center window (80% of min side) + 3×3 grid of windows at 50% of min side.
  - `iou(a, b)`, `suppress(hits)` → keep highest-confidence hit per Pokémon and drop hits overlapping a kept one (IoU > 0.3).
  - `refineCandidates(box)` → 9 windows at 75% of the box's side on a 3×3 offset grid inside it.
  - `stableNames(history)` → names seen in at least 2 of the last 3 frames.
  - `averageProbs(history)` → mean of the last 5 probability vectors (guide-box smoothing).
  - `fillTensor(rgba, out, offset)` → RGBA pixels (224×224) to normalized CHW float32 with ImageNet mean/std.
  - `softmax(row)`, `argmax(row)`.
- `wwwroot/js/live/live.js` — camera, model load/cache, frame loop, drawing, Snap. Imports `detector.js` and `result.js`.
- `wwwroot/js/result.js` — shared with the Guess page: renders top-3 bars, model badge, sprite + cry (PokéAPI), Digimon line, and ✓/✗ feedback for a given `/api/guess` response. `guess.js` is refactored to use it; upload behavior is unchanged.
- ONNX Runtime Web from jsDelivr, pinned to **1.20.1** (`ort.all.min.js`), loaded only on `/Live`.

Preprocessing matches training: each square region is cropped to its inner 87.5% (equivalent to Resize 256 → CenterCrop 224) and drawn to a 224×224 canvas.

### Models

- `training/export_onnx.py`:
  - Exports ConvNeXt with a **dynamic batch axis** (`input`/`logits` dimension 0 named `n`), still named `convnext_gen1-3.onnx`. Flask keeps using it at batch 1, unchanged.
  - Writes `convnext_gen1-3_int8.onnx` via ONNX Runtime dynamic quantization (`QuantType.QUInt8`).
  - Keeps the existing preprocessing and logits parity checks, plus: batch-of-3 logits equal three single runs (`allclose`, atol 1e-3); int8 top-1 equals fp32 top-1 on the parity images.
- GitHub release **`models-v2`** with `convnext_gen1-3.onnx`, `resnet18_gen1.onnx`, `convnext_gen1-3_int8.onnx`.
- The web app serves `convnext_gen1-3.onnx` and `convnext_gen1-3_int8.onnx` at `/models/<file>` from a directory set by config `LiveModels:Path` (default `models` under the app's base directory), outside `wwwroot` so the build does not fingerprint/compress them. Responses: `Content-Type: application/octet-stream`, `Cache-Control: public, max-age=31536000, immutable`. Only `.onnx` files are served from that directory.

### Server changes

- `Guess.Source` (`nvarchar(10)`, required, default `upload`); migration `AddGuessSource` backfills existing rows as `upload`.
- `POST /api/guess` accepts optional form field `source`: missing → `upload`; `upload` or `live` accepted; anything else → 400 `{ "error": "Unknown source." }` before calling inference.
- `StatsDto.BySource: List<GroupAccuracy>` with keys `upload` and `live`, always both present in that order; shown on `/Stats` as "By source" and returned by `/api/stats` (JSON and XML).
- Flask: no changes.

## The Live page

### Start-up

1. The page shows a **Start camera** button. The camera never starts on its own; the click also authorizes the model download.
2. The model loads from the browser Cache API (cache name `pokeguess-models-v2`) if present; otherwise it is downloaded with a progress bar labelled "Downloading model — 113 MB, once per device" (or 28 MB for the CPU fallback) and stored in the cache.
3. Backend selection: try WebGPU with `convnext_gen1-3.onnx`; on failure, WASM with `convnext_gen1-3_int8.onnx`. A status line shows "GPU (WebGPU)" or "CPU (slower)".

### Modes

| | WebGPU | No WebGPU (CPU, int8) |
|---|---|---|
| **Search whole frame** (default) | grid + refinement | disabled, with a note it needs a GPU-capable browser |
| **Guide box** | ~40 fps | the only mode, ~2 fps |

- **Search:** batch-classify `gridRegions`; keep hits ≥ 80%; `suppress`; keep names in `stableNames`; for each, up to 3 refinement passes over `refineCandidates`, accepting the best candidate while its confidence for the same Pokémon stays ≥ 80% and its side ≥ 15% of the frame's min side.
- **Guide box:** one centered square at 60% of the frame's min side; label from `averageProbs` over the last 5 frames; green with name ≥ 60%, otherwise yellow "Hold a Pokémon here".
- Boxes: thin green outlines (line width `max(2, W/450)`), label background sized to the font (font size `max(16, W/40)` px).

### Snap

- Enabled only while at least one green box is shown.
- Freezes the video and snaps the **most confident** box: crops that square region from the frozen frame, encodes JPEG (quality 0.9), and POSTs it to `/api/guess` with `source=live`.
- The result panel shows the server's answer via `result.js` (top-3, badge, sprite, cry, ✓/✗). The badge shows "Gen 1 specialist" when the legacy model answered.
- **Resume** unfreezes. Errors (400/429/503) show the server's message in the result panel; Snaps count toward the existing 10/minute rate limit.

### Unavailable / lifecycle

- No camera or permission denied: "Live mode needs a camera. You can still upload a picture" with a link to the Guess page.
- Model download failure: error message with a **Retry** button.
- `visibilitychange` to hidden: stop camera tracks and the frame loop; restart on return if the camera was running.
- Phones: rear camera (`facingMode: "environment"`) by default; **Flip camera** button only when `enumerateDevices` reports more than one video input.
- Layout follows existing mobile rules: full-width video, 16px gutter, buttons ≥ 44px.

## Error handling summary

| Failure | Behavior |
|---|---|
| No WebGPU | WASM + int8, guide-box only |
| No camera / denied | Message + link to upload |
| Model download fails | Message + Retry |
| Snap → 400/429/503 | Server's existing message in the result panel |
| Unknown `source` | 400 "Unknown source." |
| Page hidden | Camera and loop stopped; resume on return |

## Testing

**.NET (xUnit)**

- Guess with `source=live` saves `Source = "live"`; without `source` saves `"upload"`; `source=bogus` → 400 and inference not called.
- Stats: `BySource` aggregates seeded rows per source; both keys present on an empty DB with null accuracy; XML contains `BySource`.
- `/models/<file>.onnx` (temp dir via `LiveModels:Path`, small placeholder file) → 200, `application/octet-stream`, `Cache-Control` contains `immutable`; non-`.onnx` file in that dir → 404; missing file → 404.
- `/Live` → 200 and contains the Start camera button; nav contains a Live link.
- Existing 37 tests stay green.

**JavaScript (`node --test`, no dependencies)**

- `gridRegions` returns 10 squares, all inside the frame, correct sizes.
- `iou` for identical / disjoint / half-overlapping squares; `suppress` keeps one per name and drops overlapping lower-confidence hits.
- `refineCandidates` returns 9 squares at 75% inside the box.
- `stableNames` requires 2 of last 3 frames.
- `averageProbs` averages element-wise over up to 5 frames.
- `fillTensor` on a 2-pixel synthetic image produces the expected normalized values in CHW order.
- CI runs `node --test src/PokeGuess.Web/wwwroot/js/live/` (Node is preinstalled on the GitHub runner).

**Export**

- `export_onnx.py` passes its parity checks including batch-of-3 and int8 top-1 agreement.

**Manual**

- Local, desktop webcam: both modes, boxes, Snap → result + ✓/✗, Stats shows a `live` row; WebGPU disabled (`chrome://flags` or a non-WebGPU browser) → CPU fallback, guide box only.
- After deploy, phone on the live HTTPS site: rear camera, flip, second visit loads the model from cache.

## Deploy

- One-time: run the updated `export_onnx.py`, publish release `models-v2` (user approval required at that step).
- `deploy.yml`: download `models-v2` instead of `models-v1`; inference package gets `convnext_gen1-3.onnx` + `resnet18_gen1.onnx` (names unchanged); web publish gets `convnext_gen1-3.onnx` + `convnext_gen1-3_int8.onnx` in `publish/models/`.
- The post-deploy verification also checks `/models/convnext_gen1-3.onnx` returns 200 with `Content-Length` equal to the published file's size.
- `ci.yml` adds the `node --test` step to the `web` job.
- Cost note: ~113 MB per new device; Azure's first 100 GB/month of egress is free (~900 first-time visitors), then ~$0.09/GB; repeat visits use the browser cache.
- ONNX Runtime Web is pinned to an exact CDN version; its dynamically loaded WASM files come from the same pinned path.
