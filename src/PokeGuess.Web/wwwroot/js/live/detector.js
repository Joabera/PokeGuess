// Live mode detector core: pure functions, no DOM or camera (tested with node --test).
// A region is a square { x, y, s } in video pixels; a hit adds { name, conf (%), probs }.

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const SIZE = 224;

/** Search windows: one centered at 80% of the short side, plus a 3x3 grid at 50%. */
export function gridRegions(W, H) {
  const m = Math.min(W, H);
  const big = m * 0.8, s = m * 0.5;
  const regions = [{ x: (W - big) / 2, y: (H - big) / 2, s: big }];
  for (let gy = 0; gy < 3; gy++)
    for (let gx = 0; gx < 3; gx++)
      regions.push({ x: gx * (W - s) / 2, y: gy * (H - s) / 2, s });
  return regions;
}

/** Guide box: centered square at 60% of the short side. */
export function guideRegion(W, H) {
  const s = Math.min(W, H) * 0.6;
  return { x: (W - s) / 2, y: (H - s) / 2, s };
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.s, b.x + b.s) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.s, b.y + b.s) - Math.max(a.y, b.y));
  const inter = ix * iy;
  return inter / (a.s * a.s + b.s * b.s - inter);
}

/** Best hit per Pokémon; drop hits overlapping an already kept one. */
export function suppress(hits, threshold = 0.3) {
  const kept = [];
  for (const h of [...hits].sort((a, b) => b.conf - a.conf))
    if (!kept.some(k => k.name === h.name || iou(k, h) > threshold)) kept.push(h);
  return kept;
}

/** 9 windows at 75% of the box's side, on a 3x3 grid inside it. */
export function refineCandidates(box) {
  const s = box.s * 0.75, step = (box.s - s) / 2, out = [];
  for (let gy = 0; gy < 3; gy++)
    for (let gx = 0; gx < 3; gx++)
      out.push({ x: box.x + gx * step, y: box.y + gy * step, s });
  return out;
}

/** Another refinement pass is allowed while the next box stays ≥ 15% of the short side. */
export function canRefine(box, W, H) {
  return box.s * 0.75 >= Math.min(W, H) * 0.15;
}

/** Names present in at least 2 of the last 3 frames (history: Set<string>[]). */
export function stableNames(history) {
  const recent = history.slice(-3), out = new Set();
  for (const set of recent)
    for (const name of set)
      if (recent.filter(s => s.has(name)).length >= 2) out.add(name);
  return out;
}

/** Element-wise mean of the last ≤ 5 probability vectors. */
export function averageProbs(history) {
  const recent = history.slice(-5);
  return recent[0].map((_, i) => recent.reduce((sum, p) => sum + p[i], 0) / recent.length);
}

export function softmax(row) {
  const max = Math.max(...row);
  const exp = row.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

export function argmax(row) {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/** Centered 224/256 of a region: matches training's Resize(256) -> CenterCrop(224). */
export function innerCrop(r) {
  const s = r.s * SIZE / 256;
  return { x: r.x + (r.s - s) / 2, y: r.y + (r.s - s) / 2, s };
}

/** 224x224 RGBA pixels -> normalized CHW float32, written at `offset` in `out`. */
export function fillTensor(rgba, out, offset) {
  const n = SIZE * SIZE;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++)
      out[offset + c * n + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
}
