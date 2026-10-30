"""Extend the canonical class list (National Dex order) up to --max-id.

The names already in class_names_v2.txt are frozen: existing guesses, the Gen-1
specialist's name fixes and Stats depend on them. New Pokémon get their English
species name (pokemon-species/{id}, so "Giratina", not "Giratina-Altered") passed
through classlist.safe_name; the slug for sprites/cries is pokemon/{id}'s name.

Output:
  - class_names_v2.txt  : one name per line, dex order
  - pokeapi_slugs.json  : {name: pokeapi_slug}

Run from the repo root:
    python training/build_class_list.py --max-id 1025
"""
import argparse
import json
import sys
import time
import urllib.request

from classlist import merge_names, validate

SPECIES_API = "https://pokeapi.co/api/v2/pokemon-species/{id}"
POKEMON_API = "https://pokeapi.co/api/v2/pokemon/{id}"
HEADERS = {"User-Agent": "PokeGuess-dataset-builder/1.0"}


def fetch_json(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def english_name(species):
    return next(n["name"] for n in species["names"] if n["language"]["name"] == "en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-id", type=int, default=1025,
                    help="Highest National Dex id to include (1025 = end of Gen 9).")
    ap.add_argument("--out", default="models/class_names_v2.txt")
    ap.add_argument("--slugs-out", default="models/pokeapi_slugs.json")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="Delay between requests to be polite to the API.")
    args = ap.parse_args()

    with open(args.out, encoding="utf-8") as f:
        previous = [line.strip() for line in f if line.strip()]
    with open(args.slugs_out, encoding="utf-8") as f:
        previous_slugs = json.load(f)
    previous_slugs = {n: previous_slugs[n] for n in previous if n in previous_slugs}

    new_entries = []
    for pid in range(len(previous) + 1, args.max_id + 1):
        name = english_name(fetch_json(SPECIES_API.format(id=pid)))
        slug = fetch_json(POKEMON_API.format(id=pid))["name"]
        new_entries.append((name, slug))
        print(f"#{pid:>4}  {name:<20} ({slug})")
        time.sleep(args.sleep)

    names, slugs = merge_names(previous, previous_slugs, new_entries)
    errors = validate(names, slugs, previous=previous, expected=args.max_id)
    if errors:
        print("\nClass list is invalid:\n  " + "\n  ".join(errors))
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    with open(args.slugs_out, "w", encoding="utf-8") as f:
        json.dump({n: slugs[n] for n in names}, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(names)} classes -> {args.out}")
    print(f"Wrote slug map -> {args.slugs_out}")


if __name__ == "__main__":
    main()
