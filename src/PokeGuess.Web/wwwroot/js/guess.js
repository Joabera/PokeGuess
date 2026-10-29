// PokeGuess guess page: pick/drop/paste/camera -> POST /api/guess -> results + feedback.
$(function () {
  const CONFIDENCE_THRESHOLD = 60;
  const MAX_SIDE = 1024;
  const MODEL_LABELS = { "gen1-3": "Gen 1–3 model", "legacy-gen1": "Gen 1 specialist" };
  const slugs = window.POKEAPI_SLUGS || {};
  const canPlayOgg = new Audio().canPlayType("audio/ogg") !== "";

  const $drop = $("#drop"), $status = $("#status"), $preview = $("#preview");
  const $verdict = $("#verdict"), $sprite = $("#sprite"), $winner = $("#winner");
  const $badge = $("#modelBadge"), $cryBtn = $("#cryBtn"), $digimon = $("#digimon");
  const $results = $("#results"), $feedback = $("#feedback"), $actualForm = $("#actualForm");
  const $feedbackMsg = $("#feedbackMsg");
  const cry = $("#cry")[0];
  cry.volume = 0.35;

  let guessId = null;
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
    reset();
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
      .done(data => { if (isCurrent()) showResult(data); })
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

  // --- Results -----------------------------------------------------------
  function showResult(data) {
    showStatus("");
    guessId = data.guessId;
    const preds = data.predictions;
    const top = preds[0];

    $results.empty();
    preds.forEach((p, i) => {
      const pct = Math.max(0, Math.min(100, p.confidence));
      $results.append(
        $("<div class='bar-row'>").toggleClass("top", i === 0).append(
          $("<span class='name'>").text(p.name).attr("title", p.name),
          $("<div class='track'>").append($("<div class='fill'>").css("width", pct + "%")),
          $("<span class='pct'>").text(p.confidence.toFixed(1) + "%")));
    });

    $badge.text(MODEL_LABELS[data.model] || data.model);
    $verdict.removeClass("d-none");

    if (top.confidence >= CONFIDENCE_THRESHOLD) {
      $winner.text(top.name);
      loadPokemon(top.name);
    } else {
      $winner.text("");
      $digimon.text(`Are you sure this isn't a Digimon? 😕 (best guess: ${top.name}, ${top.confidence.toFixed(1)}%)`)
        .removeClass("d-none");
    }

    if (guessId !== null) $feedback.removeClass("d-none");
  }

  function loadPokemon(name) {
    const slug = slugs[name];
    if (!slug) return;
    $.getJSON(`https://pokeapi.co/api/v2/pokemon/${encodeURIComponent(slug)}`)
      .done(json => {
        const art = (json.sprites.other && json.sprites.other["official-artwork"].front_default)
          || json.sprites.front_default;
        if (art) $sprite.attr({ src: art, alt: name }).removeClass("d-none");
        const sound = json.cries && (json.cries.latest || json.cries.legacy);
        if (sound && canPlayOgg) {
          cry.src = sound;
          $cryBtn.removeClass("d-none");
          cry.play().catch(() => {}); // autoplay may be blocked; the button still works
        }
      }); // PokéAPI down: results still stand without sprite/cry
  }

  $cryBtn.on("click", () => { cry.currentTime = 0; cry.play().catch(() => {}); });

  // --- Feedback ----------------------------------------------------------
  $("#yesBtn").on("click", () => sendFeedback({ isCorrect: true }));
  $("#noBtn").on("click", () => { $actualForm.removeClass("d-none"); $("#actualName").trigger("focus"); });
  $actualForm.on("submit", e => {
    e.preventDefault();
    sendFeedback({ isCorrect: false, actualName: $("#actualName").val() });
  });

  function sendFeedback(body) {
    if (guessId === null) return;
    const $buttons = $feedback.find("button");
    $buttons.prop("disabled", true);
    $.ajax({
      url: `/api/guess/${guessId}/feedback`, method: "POST",
      data: JSON.stringify(body), contentType: "application/json",
    })
      .done(() => logged())
      .fail(xhr => {
        if (xhr.status === 409) return logged();
        $buttons.prop("disabled", false);
        $feedbackMsg.text(errorText(xhr)).addClass("text-danger");
      });
  }

  function logged() {
    $feedback.find("button, input").prop("disabled", true);
    $feedbackMsg.text("Thanks! Logged.").removeClass("text-danger");
  }

  // --- Helpers -----------------------------------------------------------
  function errorText(xhr) {
    return (xhr.responseJSON && xhr.responseJSON.error) || "Something went wrong, try again.";
  }

  function showStatus(text, isError) {
    $status.text(text).toggleClass("error", !!isError);
  }

  function reset() {
    guessId = null;
    $results.empty();
    $verdict.addClass("d-none");
    $sprite.addClass("d-none").attr("src", "");
    $cryBtn.addClass("d-none");
    $digimon.addClass("d-none").text("");
    cry.pause();
    $feedback.addClass("d-none").find("button, input").prop("disabled", false);
    $actualForm.addClass("d-none")[0].reset();
    $feedbackMsg.text("").removeClass("text-danger");
  }
});
