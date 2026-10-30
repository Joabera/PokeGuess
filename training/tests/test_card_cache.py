import json

import scrape_dataset as sd


def test_add_cards_keeps_every_dex_number():
    index = {}
    sd._add_cards(index, [
        {"nationalPokedexNumber": 25, "supertype": "Pokémon", "id": "a", "setCode": "s1"},
        {"nationalPokedexNumber": 906, "supertype": "Pokémon", "id": "b", "setCode": "s2"},
        {"nationalPokedexNumber": 25, "supertype": "Trainer", "id": "c", "setCode": "s3"},
        {"nationalPokedexNumber": None, "supertype": "Pokémon", "id": "d", "setCode": "s4"},
    ])
    assert index == {25: [("a", "s1")], 906: [("b", "s2")]}


def test_cache_round_trip(tmp_path):
    path = tmp_path / "card_index.json"
    sd._write_card_cache(str(path), {25: [("a", "s1")]})
    assert sd._read_card_cache(str(path)) == {25: [["a", "s1"]]}


def test_old_filtered_cache_is_stale(tmp_path):
    # Pre-fix caches were flat {dex: cards}, filtered to whatever --min-id asked for.
    path = tmp_path / "card_index.json"
    path.write_text(json.dumps({"387": [["a", "s1"]]}), encoding="utf-8")
    assert sd._read_card_cache(str(path)) is None


def test_missing_cache(tmp_path):
    assert sd._read_card_cache(str(tmp_path / "nope.json")) is None
