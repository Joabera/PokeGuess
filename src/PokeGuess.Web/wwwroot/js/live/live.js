// Live page: camera -> in-browser ConvNeXt (ONNX Runtime Web) -> boxes; Snap -> /api/guess.
import {
  gridRegions, guideRegion, suppress, refineCandidates, canRefine,
  stableNames, averageProbs, softmax, argmax, innerCrop, fillTensor,
} from "./detector.js";
import { createResultView, errorText } from "../result.js";

const CACHE = "pokeguess-models-v2";   // bump with the model release
const MODELS = {
  webgpu: { url: "/models/convnext_gen1-3.onnx", label: "Downloading model — 113 MB, once per device" },
  wasm: { url: "/models/convnext_gen1-3_int8.onnx", label: "Downloading model — 28 MB, once per device" },
};
const SEARCH_AT = 80, SHOW_AT = 60, SIZE = 224, PLANE = 3 * SIZE * SIZE;

const classes = window.LIVE_CLASSES;
const el = id => document.getElementById(id);
const video = el("video"), overlay = el("overlay"), octx = overlay.getContext("2d");
const crop = Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE });
const cctx = crop.getContext("2d", { willReadFrequently: true });
const view = createResultView(el("resultPanel"));

let session = null, backend = null, stream = null, facing = "environment";
let mode = "search", running = false, frozen = false, wasRunning = false;
let boxes = [], snapToken = 0;
const seen = [], probsHistory = [];

// --- Start-up -----------------------------------------------------------
el("startBtn").addEventListener("click", start);
el("retryBtn").addEventListener("click", start);

async function start() {
  show("startPanel", false); show("loadError", false); show("cameraError", false);
  try {
    await startCamera();
  } catch {
    show("cameraError", true);
    show("startPanel", true);
    return;
  }
  try {
    if (!session) await loadSession();
  } catch (e) {
    stopCamera();
    el("loadErrorText").textContent = `Couldn't load the model (${e.message}).`;
    show("loadError", true);
    return;
  }
  show("livePanel", true);
  setMode(backend === "webgpu" ? "search" : "guide");
  el("cpuNote").classList.toggle("d-none", backend === "webgpu");
  el("modeSearch").disabled = backend !== "webgpu";
  loop();
}

async function loadSession() {
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  backend = gpu ? "webgpu" : "wasm";
  const { url, label } = MODELS[backend];
  let { bytes, cached } = await fetchModel(url, label);
  try {
    session = await ort.InferenceSession.create(bytes, { executionProviders: [backend] });
  } catch (e) {
    if (!cached) throw e;
    // A bad cached copy: drop it and download once more.
    await (await caches.open(CACHE)).delete(url);
    ({ bytes } = await fetchModel(url, label));
    session = await ort.InferenceSession.create(bytes, { executionProviders: [backend] });
  }
}

// Cache API keeps the model across visits. Store only after the whole body has arrived,
// so an interrupted first download never leaves a partial model in the cache.
async function fetchModel(url, label) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true };

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  const reader = res.body.getReader(), chunks = [];
  let received = 0;
  show("progress", true);
  el("progressLabel").textContent = label;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) el("progressBar").style.width = `${(received / total * 100).toFixed(0)}%`;
  }
  show("progress", false);
  const blob = new Blob(chunks);
  await cache.put(url, new Response(blob, { headers: { "Content-Type": "application/octet-stream" } }));
  return { bytes: new Uint8Array(await blob.arrayBuffer()), cached: false };
}

// --- Camera --------------------------------------------------------------
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  show("flipBtn", cams.length > 1);
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
}

el("flipBtn").addEventListener("click", async () => {
  facing = facing === "environment" ? "user" : "environment";
  stopCamera();
  try { await startCamera(); } catch { show("cameraError", true); }
});

document.addEventListener("visibilitychange", async () => {
  if (document.hidden) {
    wasRunning = running;
    running = false;
    stopCamera();
  } else if (wasRunning) {
    wasRunning = false;
    try { await startCamera(); loop(); } catch { show("cameraError", true); }
  }
});

// --- Modes ---------------------------------------------------------------
el("modeSearch").addEventListener("click", () => setMode("search"));
el("modeGuide").addEventListener("click", () => setMode("guide"));

function setMode(m) {
  mode = m;
  seen.length = 0;
  probsHistory.length = 0;
  el("modeSearch").classList.toggle("active", m === "search");
  el("modeGuide").classList.toggle("active", m === "guide");
}

async function classify(regions) {
  const data = new Float32Array(regions.length * PLANE);
  regions.forEach((r, k) => {
    const c = innerCrop(r);
    cctx.drawImage(video, c.x, c.y, c.s, c.s, 0, 0, SIZE, SIZE);
    fillTensor(cctx.getImageData(0, 0, SIZE, SIZE).data, data, k * PLANE);
  });
  const out = await session.run({ input: new ort.Tensor("float32", data, [regions.length, 3, SIZE, SIZE]) });
  const logits = out.logits.data, C = classes.length;
  return regions.map((r, k) => {
    const probs = softmax(Array.from(logits.subarray(k * C, (k + 1) * C)));
    const best = argmax(probs);
    return { ...r, name: classes[best], conf: probs[best] * 100, probs };
  });
}

async function searchStep(W, H) {
  const hits = (await classify(gridRegions(W, H))).filter(h => h.conf >= SEARCH_AT);
  const kept = suppress(hits);
  seen.push(new Set(kept.map(k => k.name)));
  if (seen.length > 3) seen.shift();
  const stable = stableNames(seen);
  const out = [];
  for (let box of kept.filter(k => stable.has(k.name))) {
    const idx = classes.indexOf(box.name);
    for (let pass = 0; pass < 3 && canRefine(box, W, H); pass++) {
      const scored = (await classify(refineCandidates(box))).map(h => ({ ...h, name: box.name, conf: h.probs[idx] * 100 }));
      const best = scored.reduce((a, b) => (b.conf > a.conf ? b : a));
      if (best.conf < SEARCH_AT) break;
      box = best;
    }
    out.push({ ...box, ok: true });
  }
  return out;
}

async function guideStep(W, H) {
  const r = guideRegion(W, H);
  const [hit] = await classify([r]);
  probsHistory.push(hit.probs);
  if (probsHistory.length > 5) probsHistory.shift();
  const avg = averageProbs(probsHistory), best = argmax(avg), conf = avg[best] * 100;
  return [{ ...r, name: classes[best], conf, ok: conf >= SHOW_AT }];
}

// --- Loop + drawing ------------------------------------------------------
async function loop() {
  if (running) return;
  running = true;
  let frames = 0, last = performance.now();
  while (running) {
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && !frozen) {
      boxes = mode === "search" ? await searchStep(W, H) : await guideStep(W, H);
      if (!frozen) draw(W, H);
      el("snapBtn").disabled = !boxes.some(b => b.ok);
      frames++;
    }
    const now = performance.now();
    if (now - last > 1000) {
      el("liveStatus").textContent = `${backend === "webgpu" ? "GPU (WebGPU)" : "CPU (slower)"} · ${(frames * 1000 / (now - last)).toFixed(1)} fps`;
      frames = 0; last = now;
    }
    await new Promise(r => requestAnimationFrame(r));
  }
}

function draw(W, H) {
  overlay.width = W; overlay.height = H;
  const fontPx = Math.round(Math.max(16, W / 40));
  octx.lineWidth = Math.max(2, W / 450);
  octx.font = `600 ${fontPx}px system-ui`;
  for (const b of boxes) {
    const color = b.ok ? "#22c55e" : "#fbbf24";
    const label = b.ok ? `${b.name} ${b.conf.toFixed(0)}%` : "Hold a Pokémon here";
    const tw = octx.measureText(label).width + 12, th = fontPx + 10;
    octx.strokeStyle = color;
    octx.strokeRect(b.x, b.y, b.s, b.s);
    octx.fillStyle = color;
    octx.fillRect(b.x, Math.max(0, b.y - th), tw, th);
    octx.fillStyle = "#0f172a";
    octx.fillText(label, b.x + 6, Math.max(th - 8, b.y - 8));
  }
}

// --- Snap ----------------------------------------------------------------
el("snapBtn").addEventListener("click", () => {
  const target = boxes.filter(b => b.ok).sort((a, b) => b.conf - a.conf)[0];
  if (!target) return;
  frozen = true;
  video.pause();
  const token = ++snapToken;
  show("snapBtn", false); show("resumeBtn", true);

  const side = Math.min(Math.round(target.s), 1024);
  const snapCanvas = Object.assign(document.createElement("canvas"), { width: side, height: side });
  snapCanvas.getContext("2d").drawImage(video, target.x, target.y, target.s, target.s, 0, 0, side, side);
  snapCanvas.toBlob(blob => {
    if (token !== snapToken) return;
    const form = new FormData();
    form.append("file", blob, "snap.jpg");
    form.append("source", "live");
    $.ajax({ url: "/api/guess", method: "POST", data: form, processData: false, contentType: false })
      .done(data => { if (token === snapToken) view.show(data); })
      .fail(xhr => { if (token === snapToken) view.showError(errorText(xhr)); });
  }, "image/jpeg", 0.9);
});

el("resumeBtn").addEventListener("click", () => {
  snapToken++; // in-flight Snap answers are now stale
  view.reset();
  frozen = false;
  video.play();
  show("resumeBtn", false); show("snapBtn", true);
});

function show(id, visible) {
  el(id).classList.toggle("d-none", !visible);
}
