import pytest

from classlist import merge_names, safe_name, validate


@pytest.mark.parametrize("raw,safe", [
    ("Type: Null", "Type Null"),
    ("Mime Jr.", "Mime Jr"),
    ("A  B", "A B"),
    ('Bad<>:"/\\|?*Name', "BadName"),
    ("Farfetch'd", "Farfetch'd"),
    ("Flabébé", "Flabébé"),
    ("Porygon-Z", "Porygon-Z"),
    ("Mr. Mime", "Mr. Mime"),
])
def test_safe_name(raw, safe):
    assert safe_name(raw) == safe


def test_merge_keeps_previous_and_sanitizes_new():
    names, slugs = merge_names(["Bulbasaur"], {"Bulbasaur": "bulbasaur"}, [("Type: Null", "type-null")])
    assert names == ["Bulbasaur", "Type Null"]
    assert slugs == {"Bulbasaur": "bulbasaur", "Type Null": "type-null"}


def test_validate_ok():
    names = ["Bulbasaur", "Ivysaur", "Turtwig"]
    slugs = {n: n.lower() for n in names}
    assert validate(names, slugs, previous=names[:2], expected=3) == []


def test_validate_rejects_changed_previous():
    errors = validate(["Bulbasaur2", "Ivysaur"], {"Bulbasaur2": "b", "Ivysaur": "i"},
                      previous=["Bulbasaur", "Ivysaur"], expected=2)
    assert any("#1" in e for e in errors)


def test_validate_rejects_duplicate_case_insensitive():
    errors = validate(["Eevee", "eevee"], {"Eevee": "eevee", "eevee": "eevee"}, previous=[], expected=2)
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_rejects_unsafe_long_missing_slug_and_count():
    long_name = "X" * 41
    names = ["Type: Null", long_name, "Noslug"]
    slugs = {"Type: Null": "type-null", long_name: "x"}
    errors = validate(names, slugs, previous=[], expected=4)
    joined = " | ".join(errors).lower()
    assert len(errors) == 4
    assert "unsafe" in joined and "40" in joined and "slug" in joined and "expected 4" in joined
