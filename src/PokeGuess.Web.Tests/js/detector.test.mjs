// Run: node --test src/PokeGuess.Web.Tests/js/
import test from "node:test";
import assert from "node:assert/strict";
import {
  gridRegions, guideRegion, iou, suppress, refineCandidates, canRefine,
  stableNames, averageProbs, softmax, argmax, innerCrop, fillTensor,
} from "../../PokeGuess.Web/wwwroot/js/live/detector.js";

const inside = (r, W, H) => r.x >= 0 && r.y >= 0 && r.x + r.s <= W + 1e-9 && r.y + r.s <= H + 1e-9;
const hit = (name, conf, x, y, s) => ({ name, conf, x, y, s });

test("gridRegions: 1 center window at 80% + 3x3 at 50%, inside a landscape frame", () => {
  const r = gridRegions(1280, 720);
  assert.equal(r.length, 10);
  assert.equal(r[0].s, 576);
  assert.deepEqual([r[0].x, r[0].y], [352, 72]);
  for (const g of r.slice(1)) assert.equal(g.s, 360);
  assert.ok(r.every(g => inside(g, 1280, 720)));
});

test("gridRegions stays inside portrait and low-res frames", () => {
  for (const [W, H] of [[720, 1280], [320, 240]]) assert.ok(gridRegions(W, H).every(g => inside(g, W, H)), `${W}x${H}`);
});

test("guideRegion is a centered square at 60% of the short side", () => {
  assert.deepEqual(guideRegion(1280, 720), { x: 424, y: 144, s: 432 });
  assert.ok(inside(guideRegion(720, 1280), 720, 1280));
});

test("iou", () => {
  const a = { x: 0, y: 0, s: 10 };
  assert.equal(iou(a, a), 1);
  assert.equal(iou(a, { x: 20, y: 20, s: 10 }), 0);
  assert.ok(Math.abs(iou(a, { x: 5, y: 0, s: 10 }) - 1 / 3) < 1e-9);
});

test("suppress keeps one per name and drops overlapping lower-confidence hits", () => {
  const kept = suppress([
    hit("Eevee", 85, 0, 0, 100), hit("Eevee", 92, 300, 300, 100),
    hit("Pikachu", 88, 310, 300, 100), hit("Mew", 81, 700, 0, 100),
  ]);
  assert.deepEqual(kept.map(k => [k.name, k.conf]), [["Eevee", 92], ["Mew", 81]]);
});

test("refineCandidates: 9 windows at 75% inside the box", () => {
  const c = refineCandidates({ x: 100, y: 100, s: 400 });
  assert.equal(c.length, 9);
  assert.ok(c.every(r => r.s === 300 && r.x >= 100 && r.y >= 100 && r.x + r.s <= 500 && r.y + r.s <= 500));
  assert.ok(c.some(r => r.x === 100 && r.y === 100));
  assert.ok(c.some(r => r.x === 200 && r.y === 200));
});

test("canRefine stops below 15% of the short side", () => {
  assert.equal(canRefine({ x: 0, y: 0, s: 48 }, 320, 240), true);   // 0.75*48 = 36 = 15% of 240
  assert.equal(canRefine({ x: 0, y: 0, s: 40 }, 320, 240), false);  // 30 < 36
  assert.equal(canRefine({ x: 0, y: 0, s: 400 }, 1280, 720), true);
});

test("stableNames needs 2 of the last 3 frames", () => {
  assert.deepEqual([...stableNames([new Set(["A"]), new Set(["B"]), new Set(["A"])])], ["A"]);
  assert.deepEqual([...stableNames([new Set(["A"]), new Set(["B"]), new Set(["C"])])], []);
  assert.deepEqual([...stableNames([new Set(["A"]), new Set(["A"]), new Set(["B"]), new Set(["C"])])], []);
});

test("averageProbs uses only the last 5 frames", () => {
  const h = [[1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]];
  assert.deepEqual(averageProbs(h), [0, 1]);
  assert.deepEqual(averageProbs([[1, 0], [0, 1]]), [0.5, 0.5]);
});

test("softmax and argmax", () => {
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-12);
  assert.equal(argmax([1, 3, 2]), 1);
});

test("innerCrop takes the centered 224/256 of the region (Resize 256 -> CenterCrop 224)", () => {
  assert.deepEqual(innerCrop({ x: 0, y: 0, s: 256 }), { x: 16, y: 16, s: 224 });
});

test("fillTensor normalizes RGBA into CHW planes", () => {
  const n = 224 * 224, rgba = new Uint8ClampedArray(n * 4);
  rgba.set([255, 255, 255, 255], 0);   // pixel 0 white; pixel 1 black (zeros)
  const out = new Float32Array(2 + 3 * n);
  fillTensor(rgba, out, 2);
  const close = (a, b) => Math.abs(a - b) < 1e-6;
  assert.ok(close(out[2], (1 - 0.485) / 0.229));          // R, pixel 0
  assert.ok(close(out[3], (0 - 0.485) / 0.229));          // R, pixel 1
  assert.ok(close(out[2 + n], (1 - 0.456) / 0.224));      // G, pixel 0
  assert.ok(close(out[2 + 2 * n], (1 - 0.406) / 0.225));  // B, pixel 0
  assert.equal(out[0], 0);                                  // before offset untouched
});
