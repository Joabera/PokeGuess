// Renders a POST /api/guess answer (bars, badge, sprite + cry, Digimon line) and its
// ✓/✗ feedback inside the _ResultPanel partial. Shared by the Guess and Live pages.
const CONFIDENCE_THRESHOLD = 60;
const MODEL_LABELS = { "gen1-3": "Gen 1–3 model", "legacy-gen1": "Gen 1 specialist" };
const canPlayOgg = new Audio().canPlayType("audio/ogg") !== "";

export function errorText(xhr) {
  return (xhr.responseJSON && xhr.responseJSON.error) || "Something went wrong, try again.";
}

export function createResultView(root) {
  const $root = $(root), find = sel => $root.find(sel);
  const $verdict = find("#verdict"), $sprite = find("#sprite"), $winner = find("#winner");
  const $badge = find("#modelBadge"), $cryBtn = find("#cryBtn"), $digimon = find("#digimon");
  const $results = find("#results"), $feedback = find("#feedback"), $actualForm = find("#actualForm");
  const $feedbackMsg = find("#feedbackMsg"), $error = find("#resultError");
  const cry = find("#cry")[0];
  const slugs = window.POKEAPI_SLUGS || {};
  cry.volume = 0.35;

  let guessId = null;
  let generation = 0; // bumped by reset(): late responses for an old guess are ignored

  $cryBtn.on("click", () => { cry.currentTime = 0; cry.play().catch(() => {}); });
  find("#yesBtn").on("click", () => sendFeedback({ isCorrect: true }));
  find("#noBtn").on("click", () => { $actualForm.removeClass("d-none"); find("#actualName").trigger("focus"); });
  $actualForm.on("submit", e => {
    e.preventDefault();
    sendFeedback({ isCorrect: false, actualName: find("#actualName").val() });
  });

  function show(data) {
    reset();
    guessId = data.guessId;
    const preds = data.predictions, top = preds[0];

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
      loadPokemon(top.name, generation);
    } else {
      $digimon.text(`Are you sure this isn't a Digimon? 😕 (best guess: ${top.name}, ${top.confidence.toFixed(1)}%)`)
        .removeClass("d-none");
    }

    if (guessId !== null) $feedback.removeClass("d-none");
  }

  function showError(message) {
    reset();
    $error.text(message).removeClass("d-none");
  }

  function loadPokemon(name, gen) {
    const slug = slugs[name];
    if (!slug) return;
    $.getJSON(`https://pokeapi.co/api/v2/pokemon/${encodeURIComponent(slug)}`)
      .done(json => {
        if (gen !== generation) return;
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

  function sendFeedback(body) {
    if (guessId === null) return;
    const gen = generation;
    const $buttons = $feedback.find("button");
    $buttons.prop("disabled", true);
    $.ajax({
      url: `/api/guess/${guessId}/feedback`, method: "POST",
      data: JSON.stringify(body), contentType: "application/json",
    })
      .done(() => { if (gen === generation) logged(); })
      .fail(xhr => {
        if (gen !== generation) return;
        if (xhr.status === 409) return logged();
        $buttons.prop("disabled", false);
        $feedbackMsg.text(errorText(xhr)).addClass("text-danger");
      });
  }

  function logged() {
    $feedback.find("button, input").prop("disabled", true);
    $feedbackMsg.text("Thanks! Logged.").removeClass("text-danger");
  }

  function reset() {
    generation++;
    guessId = null;
    $error.addClass("d-none").text("");
    $results.empty();
    $verdict.addClass("d-none");
    $winner.text("");
    $badge.text("");
    $sprite.addClass("d-none").attr("src", "");
    $cryBtn.addClass("d-none");
    $digimon.addClass("d-none").text("");
    cry.pause();
    $feedback.addClass("d-none").find("button, input").prop("disabled", false);
    $actualForm.addClass("d-none")[0].reset();
    $feedbackMsg.text("").removeClass("text-danger");
  }

  return { show, showError, reset };
}
