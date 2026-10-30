// PokeGuess guess page: pick/drop/paste/camera -> POST /api/guess -> result panel (result.js).
import { createResultView, errorText } from "./result.js";

const MAX_SIDE = 1024;
const view = createResultView(document.getElementById("resultPanel"));
const $drop = $("#drop"), $status = $("#status"), $preview = $("#preview");

let previewUrl = null;
let currentRequest = 0; // only the latest upload may update the page

if (window.matchMedia("(pointer: coarse)").matches) $("#cameraBtn").removeClass("d-none");

// --- Picking an image ---------------------------------------------------
// Native .click(), not jQuery .trigger(): the inputs live inside #drop, and trigger()
// runs #drop's click handler again on the way up, recursing until the stack overflows.
const openPicker = id => document.getElementById(id).click();
$("#fileInput, #cameraInput").on("click", e => e.stopPropagation());
$("#pickBtn").on("click", e => { e.stopPropagation(); openPicker("fileInput"); });
$("#cameraBtn").on("click", e => { e.stopPropagation(); openPicker("cameraInput"); });
$drop.on("click", () => openPicker("fileInput"));
$drop.on("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker("fileInput"); } });
$("#fileInput, #cameraInput").on("change", function () {
  if (this.files[0]) handleFile(this.files[0]);
  this.value = "";
});

$drop.on("dragover", e => { e.preventDefault(); $drop.addClass("drag"); });
$drop.on("dragleave drop", () => $drop.removeClass("drag"));
$drop.on("drop", e => {
  e.preventDefault();
  const file = e.originalEvent.dataTransfer.files[0];
  if (file) handleFile(file);
});

$(document).on("paste", e => {
  const items = (e.originalEvent.clipboardData || {}).items || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) { handleFile(item.getAsFile()); return; }
  }
});

// --- Upload ------------------------------------------------------------
async function handleFile(file) {
  if (!file.type.startsWith("image/")) return showStatus("Couldn't read that image.", true);
  const request = ++currentRequest;
  const isCurrent = () => request === currentRequest;
  view.reset();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $preview.attr("src", previewUrl).show();

  $drop.addClass("busy");
  showStatus("Thinking…");
  let upload;
  try {
    upload = await downscale(file);
  } catch {
    upload = file; // let the server decide if it can't be decoded here (e.g. HEIC)
  }
  if (!isCurrent()) return;

  const form = new FormData();
  form.append("file", upload, upload.name || "image");
  $.ajax({ url: "/api/guess", method: "POST", data: form, processData: false, contentType: false })
    .done(data => { if (isCurrent()) { showStatus(""); view.show(data); } })
    .fail(xhr => { if (isCurrent()) showStatus(errorText(xhr), true); })
    .always(() => { if (isCurrent()) $drop.removeClass("busy"); });
}

// Phone photos are large: shrink anything over MAX_SIDE. PNG stays PNG (keeps sketches crisp).
function downscale(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = MAX_SIDE / Math.max(img.width, img.height);
      if (scale >= 1) return resolve(file);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      const type = file.type === "image/png" ? "image/png" : "image/jpeg";
      canvas.toBlob(blob => blob
        ? resolve(new File([blob], file.name || "image", { type }))
        : reject(new Error("encode failed")), type, 0.9);
    };
    img.onerror = reject;
    img.src = previewUrl;
  });
}

function showStatus(text, isError) {
  $status.text(text).toggleClass("error", !!isError);
}
