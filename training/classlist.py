"""Class-name rules shared by the class-list builder and the scraper.

One canonical name per Pokémon is used everywhere: class file, dataset folder,
model output, database and web pages. Names must therefore be valid Windows folder
names exactly as written (ImageFolder uses folder names as labels).
"""
import re

MAX_LEN = 40  # Guesses.PredictedName / ActualName column size
_ILLEGAL = re.compile(r'[<>:"/\\|?*]')


def safe_name(name):
    """'Type: Null' -> 'Type Null', 'Mime Jr.' -> 'Mime Jr' (Windows drops trailing dots)."""
    name = _ILLEGAL.sub("", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip().rstrip(". ").strip()


def merge_names(previous, previous_slugs, new_entries):
    """Keep `previous` names untouched; append sanitized new (species_name, slug) entries."""
    names = list(previous)
    slugs = dict(previous_slugs)
    for species_name, slug in new_entries:
        name = safe_name(species_name)
        names.append(name)
        slugs[name] = slug
    return names, slugs


def validate(names, slugs, previous, expected):
    """Return a list of problems; empty means the class list is usable."""
    errors = []
    if len(names) != expected:
        errors.append(f"expected {expected} names, got {len(names)}")
    for i, old in enumerate(previous):
        if i >= len(names) or names[i] != old:
            errors.append(f"#{i + 1} changed: {old!r} -> {names[i] if i < len(names) else None!r}")
    seen = {}
    for name in names:
        key = name.casefold()
        if key in seen:
            errors.append(f"duplicate name (case-insensitive): {seen[key]!r} / {name!r}")
        seen[key] = name
        if safe_name(name) != name:
            errors.append(f"unsafe name {name!r} (safe: {safe_name(name)!r})")
        if len(name) > MAX_LEN:
            errors.append(f"name longer than {MAX_LEN} characters: {name!r}")
        if not slugs.get(name):
            errors.append(f"no slug for {name!r}")
    return errors
