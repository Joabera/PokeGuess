"""Build a clean, canonical class list for Gen 1 + Gen 2 (Pokedex #1-251).

Pulls names from PokeAPI in National Dex order. This replaces the old
hand-made class_names.txt, which had duplicate/odd spellings for the same
Pokemon (MrMime/Mr. Mime/Mime, Nidoran1/Nidoran2, Farfetch'd/Farfetchd).

Output:
  - class_names_v2.txt  : one display name per line, dex order (#1..#251)
  - pokeapi_slugs.json  : {display_name: pokeapi_slug} for the scraper/app

Run:
    python build_class_list.py --max-id 251
"""
import argparse
import json
import time
import urllib.request

API = "https://pokeapi.co/api/v2/pokemon/{id}"
HEADERS = {"User-Agent": "PokeGuess-dataset-builder/1.0"}


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def display_name(slug):
    """Turn a PokeAPI slug into a friendly display name."""
    special = {
        "nidoran-f": "Nidoran-F",
        "nidoran-m": "Nidoran-M",
        "mr-mime": "Mr. Mime",
        "farfetchd": "Farfetch'd",
        "ho-oh": "Ho-Oh",
    }
    if slug in special:
        return special[slug]
    return "-".join(part.capitalize() for part in slug.split("-"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-id", type=int, default=251,
                    help="Highest National Dex id to include (251 = end of Gen 2).")
    ap.add_argument("--out", default="models/class_names_v2.txt")
    ap.add_argument("--slugs-out", default="models/pokeapi_slugs.json")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="Delay between requests to be polite to the API.")
    args = ap.parse_args()

    names, slug_map = [], {}
    for pid in range(1, args.max_id + 1):
        data = fetch_json(API.format(id=pid))
        slug = data["name"]
        name = display_name(slug)
        names.append(name)
        slug_map[name] = slug
        print(f"#{pid:>3}  {name:<14} ({slug})")
        time.sleep(args.sleep)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    with open(args.slugs_out, "w", encoding="utf-8") as f:
        json.dump(slug_map, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(names)} classes -> {args.out}")
    print(f"Wrote slug map -> {args.slugs_out}")


if __name__ == "__main__":
    main()
