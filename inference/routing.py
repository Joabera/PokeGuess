"""Two-model routing: the Gen 1-9 ConvNeXt picks the generation; for Gen 1 it
defers to the legacy ResNet-18 (trained on more Gen 1 data) when that model is
confident enough."""

# Legacy class list has odd spellings; map them to the canonical names.
LEGACY_FIX = {
    "Farfetchd": "Farfetch'd", "Mime": "Mr. Mime", "MrMime": "Mr. Mime",
    "Nidoran1": "Nidoran-F", "Nidoran2": "Nidoran-M",
}


def route(new_preds, legacy_predict, gen1_names, legacy_trust_pct):
    """new_preds/legacy results are [(name, confidence_pct)], best first.
    legacy_predict is only called when the new model's top-1 is Gen 1.
    Returns (predictions, "legacy-gen1" | "gen1-9")."""
    if new_preds[0][0] in gen1_names:
        legacy = [(LEGACY_FIX.get(n, n), c) for n, c in legacy_predict()]
        if legacy[0][1] >= legacy_trust_pct:
            return legacy, "legacy-gen1"
    return new_preds, "gen1-9"
