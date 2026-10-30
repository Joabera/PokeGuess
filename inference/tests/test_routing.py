import pytest

from routing import route

GEN1 = {"Pikachu", "Mr. Mime", "Nidoran-F"}


def never_called():
    raise AssertionError("legacy model should not run")


def test_non_gen1_never_calls_legacy():
    new = [("Treecko", 90.0), ("Grovyle", 5.0)]
    preds, model = route(new, never_called, GEN1, 55.0)
    assert model == "gen1-9"
    assert preds == new


def test_gen1_legacy_confident_wins():
    legacy = [("Pikachu", 55.0), ("Raichu", 30.0)]
    preds, model = route([("Pikachu", 70.0)], lambda: legacy, GEN1, 55.0)
    assert model == "legacy-gen1"
    assert preds == legacy


def test_gen1_legacy_below_trust_uses_new():
    new = [("Pikachu", 70.0), ("Raichu", 20.0)]
    preds, model = route(new, lambda: [("Pikachu", 54.99)], GEN1, 55.0)
    assert model == "gen1-9"
    assert preds == new


@pytest.mark.parametrize("raw,fixed", [
    ("MrMime", "Mr. Mime"), ("Mime", "Mr. Mime"), ("Farfetchd", "Farfetch'd"),
    ("Nidoran1", "Nidoran-F"), ("Nidoran2", "Nidoran-M"),
])
def test_legacy_names_are_fixed(raw, fixed):
    preds, model = route([("Pikachu", 70.0)], lambda: [(raw, 90.0), ("Pikachu", 5.0)], GEN1, 55.0)
    assert model == "legacy-gen1"
    assert [n for n, _ in preds] == [fixed, "Pikachu"]
