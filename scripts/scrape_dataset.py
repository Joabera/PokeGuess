"""PokeGuess dataset-building agent.

Builds an ImageFolder-style training set for Pokemon classification by
combining clean PokeAPI imagery with scraped web images, then cleaning the
result. Designed to be run stage-by-stage and resumable.

Pipeline (each stage is independent and idempotent):
  1. seed   - download guaranteed-correct images from PokeAPI per Pokemon
              (official artwork, default/shiny sprites, Home/Showdown sprites)
  2. scrape - crawl Bing (icrawler) + DuckDuckGo (ddgs) image search for
              variety. Google is intentionally not used: it blocks scrapers
              and icrawler's Google parser is broken.
  3. dedup  - drop near-duplicate images via perceptual hash (requires `imagehash`)
  4. clip   - drop off-topic / wrong-Pokemon images via CLIP zero-shot
              (requires `open_clip_torch`)

Output layout (consumable directly by torchvision.datasets.ImageFolder):
  data/raw/<Display Name>/seed_*.png
                         /bing_*.jpg
                         ...

Examples:
  # Just the clean PokeAPI foundation for every class:
  python scrape_dataset.py seed

  # Scrape ~80 web images per Pokemon for Gen 2 only (ids 152-251):
  python scrape_dataset.py scrape --per-class 80 --min-id 152

  # Clean everything:
  python scrape_dataset.py dedup
  python scrape_dataset.py clip --threshold 0.22

  # Or run the whole thing:
  python scrape_dataset.py all --per-class 80
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

from tqdm import tqdm
import urllib.request
import urllib.parse

# Pillow warns about palette PNGs with byte transparency (common in sprites/
# cards); harmless for our purposes and it clutters the progress bars.
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Windows consoles default to cp1252, which crashes when we print scraped URLs
# containing non-Latin characters. Force UTF-8 and never let a print abort a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths are relative to the repo root (run scripts as `python scripts/<name>.py`).
DATA_ROOT = "data/raw"
CLASS_FILE = "models/class_names_v2.txt"
SLUG_FILE = "models/pokeapi_slugs.json"
HEADERS = {"User-Agent": "PokeGuess-dataset-builder/1.0"}
# pokemoncard.io 403s non-browser agents.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_classes(min_id=1, max_id=251, only=None):
    """Return list of (dex_index, display_name, pokeapi_slug)."""
    names = [l.strip() for l in open(CLASS_FILE, encoding="utf-8") if l.strip()]
    slugs = json.load(open(SLUG_FILE, encoding="utf-8"))
    out = []
    for i, name in enumerate(names, start=1):
        if i < min_id or i > max_id:
            continue
        if only and name.lower() not in {o.lower() for o in only}:
            continue
        out.append((i, name, slugs.get(name, name.lower())))
    return out


def class_dir(name):
    d = os.path.join(DATA_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


# Hosts that serve adult / junk content. Any URL on these is rejected outright,
# regardless of search engine. Substring match against the hostname.
BLOCKED_HOSTS = (
    "xhcdn", "xhamster", "pornhub", "phncdn", "xvideos", "xnxx", "redtube",
    "rule34", "e621", "e-hentai", "hentai", "nhentai", "gelbooru", "danbooru",
    "sankaku", "porn", "nsfw", "adult", "xxx", "redgifs", "imagefap",
)


def _is_blocked(url):
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(b in host for b in BLOCKED_HOSTS)


def download(url, path):
    if not url or _is_blocked(url):
        return False
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        safe = str(e).encode("ascii", "replace").decode("ascii")
        tqdm.write(f"    ! download failed: {safe}")
        return False


# --------------------------------------------------------------------------- #
# Stage 1: PokeAPI seed
# --------------------------------------------------------------------------- #
def collect_sprite_urls(sprites):
    """Pull every usable static image URL out of a PokeAPI sprites block."""
    urls = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "versions":  # huge nested game-by-game tree; skip
                    continue
                walk(v)
        elif isinstance(node, str) and node.startswith("http") and node.endswith(".png"):
            urls.append(node)

    walk(sprites)
    # de-dup, keep order
    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def stage_seed(classes, sleep=0.05):
    total = 0
    bar = tqdm(classes, desc="seed  ", unit="cls")
    for dex, name, slug in bar:
        d = class_dir(name)
        try:
            data = fetch_json(f"https://pokeapi.co/api/v2/pokemon/{slug}")
        except Exception as e:
            tqdm.write(f"#{dex} {name}: API error {e}")
            continue
        urls = collect_sprite_urls(data.get("sprites", {}))
        saved = 0
        for j, url in enumerate(urls):
            path = os.path.join(d, f"seed_{j:02d}.png")
            if os.path.exists(path):
                saved += 1
                continue
            if download(url, path):
                saved += 1
        total += saved
        bar.set_postfix_str(f"{name} (+{saved}, {total} total)")
        time.sleep(sleep)
    print(f"seed done, {total} images")


# --------------------------------------------------------------------------- #
# Stage 1b: Pokemon TCG card art (pokemoncard.io)
# --------------------------------------------------------------------------- #
CARD_INDEX_CACHE = "card_index.json"
CARD_DB = "https://pokemoncard.io/api/cards/database?page={page}"
CARD_IMG = "https://images.pokemoncard.io/images/{set}/{id}{suffix}.png"


def _download_card_cropped(card_id, set_code, path, crop_frac):
    """Download a card (hi-res, falling back to standard) and keep the top
    `crop_frac` (the artwork), discarding the lower attack-text / stats region."""
    from PIL import Image
    import io
    urls = [CARD_IMG.format(set=set_code, id=card_id, suffix="_hires"),
            CARD_IMG.format(set=set_code, id=card_id, suffix="")]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = img.size
            img.crop((0, 0, w, int(h * crop_frac))).save(path)
            return True
        except Exception:
            continue
    return False


def _build_card_index(wanted_dex, refresh=False, sleep=0.4):
    """Paginate the entire pokemoncard.io catalogue ONCE and bucket every
    Pokemon card by National Dex number. Cached to disk so it only runs once.

    The DB endpoint has no working name filter, so a full scan is the only way
    to get *all* cards per Pokemon. Polite pacing + 429 backoff keeps us under
    the rate limit. Returns {dex: [(id, setCode), ...]}.
    """
    if os.path.exists(CARD_INDEX_CACHE) and not refresh:
        raw = json.load(open(CARD_INDEX_CACHE, encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    index = defaultdict(list)
    page, last_page = 1, None
    completed = False
    bar = tqdm(desc="cards:index", unit="pg")
    while True:
        j = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(
                    CARD_DB.format(page=page),
                    headers={"User-Agent": BROWSER_UA, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    j = json.load(r)
                break
            except Exception as e:
                # Retry ANY network hiccup (429, SSL timeout, reset, ...) with
                # backoff -- a single transient error must not abort the scan.
                wait = (4 * (attempt + 1)) if "429" in str(e) else (1.5 * (attempt + 1))
                if attempt < 7:
                    time.sleep(wait)
                    continue
                tqdm.write(f"    ! card index page {page} gave up: {e}")
        if j is None:
            break  # page failed after all retries; abort without caching
        last_page = last_page or j.get("last_page", 1)
        if bar.total is None:
            bar.total = last_page
        for c in j.get("data", []):
            dex = c.get("nationalPokedexNumber")
            if (dex in wanted_dex and c.get("supertype") == "Pokémon"
                    and c.get("id") and c.get("setCode")):
                index[dex].append((c["id"], c["setCode"]))
        bar.update(1)
        bar.set_postfix_str(f"{sum(len(v) for v in index.values())} cards")
        if page >= last_page:
            completed = True
            break
        page += 1
        time.sleep(sleep)
    bar.close()

    # Only cache a COMPLETE scan, so a partial run never poisons later runs.
    if completed:
        json.dump({str(k): v for k, v in index.items()},
                  open(CARD_INDEX_CACHE, "w", encoding="utf-8"))
    else:
        tqdm.write(f"    ! card scan incomplete (reached page {page}/{last_page}); "
                   "not caching. Re-run to resume.")
    return dict(index)


def stage_cards(classes, crop_frac=0.52, cap=0, refresh=False):
    """Collect ALL official TCG card art per Pokemon, cropped to the top
    `crop_frac` (the illustration). Official art => clean, correctly labeled,
    zero NSFW risk. `cap=0` means unlimited."""
    wanted = {dex: name for dex, name, _ in classes}
    index = _build_card_index(set(wanted), refresh=refresh)

    saved_total = 0
    bar = tqdm(classes, desc="cards ", unit="cls")
    for dex, name, _ in bar:
        d = class_dir(name)
        cards = index.get(dex, [])
        if cap:
            cards = cards[:cap]
        saved = 0
        for card_id, set_code in cards:
            path = os.path.join(d, f"card_{card_id}.png")
            if os.path.exists(path):
                saved += 1
                continue
            if _download_card_cropped(card_id, set_code, path, crop_frac):
                saved += 1
        saved_total += saved
        bar.set_postfix_str(f"{name} ({len(cards)} cards, {saved_total} total)")
    print(f"cards done, saved {saved_total} cropped card images total")


# --------------------------------------------------------------------------- #
# Stage 2: web scrape (optional dep: icrawler)
# --------------------------------------------------------------------------- #
def _scrape_ddg(query, dest_dir, max_num):
    """DuckDuckGo image scrape (icrawler has no DDG crawler; Google is blocked)."""
    from ddgs import DDGS
    n = 0
    base = len(os.listdir(dest_dir))
    try:
        # safesearch="on" = strict; the strongest filter DDG offers.
        results = DDGS().images(query, safesearch="on", max_results=max_num * 3)
    except Exception as e:
        tqdm.write(f"    ! ddg search error '{query}': {e}")
        return 0
    for r in results:
        if n >= max_num:
            break
        url = r.get("image")
        if not url or _is_blocked(url):
            continue
        ext = os.path.splitext(url.split("?")[0])[1].lower() if url else ""
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        path = os.path.join(dest_dir, f"ddg_{base + n:05d}{ext}")
        if download(url, path):
            n += 1
    return n


def stage_scrape(classes, per_class=120, engines=("bing", "ddg")):
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        sys.exit("Stage 'scrape' needs icrawler:  pip install icrawler")

    ICRAWLERS = {"bing": BingImageCrawler}
    bar = tqdm(classes, desc=f"scrape({'+'.join(engines)})", unit="cls")
    for dex, name, _ in bar:
        bar.set_postfix_str(name)
        d = class_dir(name)
        existing = len([f for f in os.listdir(d) if not f.startswith("seed_")])
        if existing >= per_class:
            continue

        # Queries are deliberately specific (official art / game / TCG / sketch)
        # to steer away from NSFW fan content. Bare "{name} pokemon" and
        # "anime screenshot" were removed -- they were the main junk magnets.
        queries = [
            f"{name} pokemon official artwork",
            f"{name} pokemon video game render",
            f"{name} pokemon trading card",
            f"{name} pokemon pencil sketch drawing",
        ]
        remaining = per_class - existing
        # Spread the shortfall across every engine x query combination.
        per_query = max(1, remaining // (len(engines) * len(queries)) + 1)
        for engine in engines:
            for q in queries:
                if engine == "ddg":
                    _scrape_ddg(q, d, per_query)
                elif engine in ICRAWLERS:
                    crawler = ICRAWLERS[engine](
                        storage={"root_dir": d},
                        downloader_threads=2,
                        log_level="CRITICAL",  # silence per-image 403/401 spam
                    )
                    # Best-effort: ask Bing for strict adult filtering. The
                    # domain blocklist + CLIP filter are the real safety net.
                    try:
                        crawler.session.cookies.set("SRCHHPGUSR", "ADLT=STRICT")
                    except Exception:
                        pass
                    try:
                        # "auto" offset continues numbering from existing files,
                        # so results from each engine accumulate without overwriting.
                        crawler.crawl(keyword=q, max_num=per_query,
                                      file_idx_offset="auto",
                                      filters={"type": "photo"})
                    except Exception as e:
                        tqdm.write(f"    ! {engine} crawl error '{q}': {e}")
        total = len([f for f in os.listdir(d) if not f.startswith("seed_")])
        bar.set_postfix_str(f"{name} ({total} scraped)")


# --------------------------------------------------------------------------- #
# Stage 3: perceptual-hash dedup (optional dep: imagehash)
# --------------------------------------------------------------------------- #
def stage_dedup(classes, hash_size=8):
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        sys.exit("Stage 'dedup' needs imagehash + Pillow:  pip install imagehash")

    removed_total = 0
    bar = tqdm(classes, desc="dedup ", unit="cls")
    for dex, name, _ in bar:
        d = class_dir(name)
        seen = {}
        removed = 0
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            try:
                with Image.open(path) as im:
                    h = str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
            except Exception:
                os.remove(path)  # unreadable image
                removed += 1
                continue
            if h in seen:
                os.remove(path)
                removed += 1
            else:
                seen[h] = fn
        removed_total += removed
        bar.set_postfix_str(f"{name} (-{removed_total} dupes)")
    print(f"dedup done, removed {removed_total} images total")


# --------------------------------------------------------------------------- #
# Stage 4: CLIP zero-shot junk filter (optional dep: open_clip_torch)
# --------------------------------------------------------------------------- #
def stage_clip(classes, threshold=0.22, model_name="ViT-B-32",
               pretrained="laion2b_s34b_b79k"):
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError:
        sys.exit("Stage 'clip' needs open_clip_torch:  pip install open_clip_torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Distractors include explicit NSFW prompts: an image is dropped if it
    # matches any distractor better than the target, OR if P(target) is low.
    # The NSFW prompts give CLIP an explicit bucket to send adult images to.
    nsfw_prompts = ["a nude person", "explicit pornography", "a sexual image"]
    distractors = ["a random photo", "text", "a screenshot of a website",
                   "a different cartoon character", "a meme"] + nsfw_prompts
    n_nsfw = len(nsfw_prompts)

    removed_total, nsfw_total = 0, 0
    bar = tqdm(classes, desc="clip  ", unit="cls")
    for dex, name, _ in bar:
        d = class_dir(name)
        prompts = [f"a picture of {name}, a Pokemon"] + distractors
        text = tokenizer(prompts).to(device)
        with torch.no_grad():
            tfeat = model.encode_text(text)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)

        removed, nsfw_hits = 0, 0
        for fn in sorted(os.listdir(d)):
            if fn.startswith("seed_"):
                continue  # trust PokeAPI seeds, never drop them
            path = os.path.join(d, fn)
            try:
                img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            except Exception:
                os.remove(path)
                removed += 1
                continue
            with torch.no_grad():
                ifeat = model.encode_image(img)
                ifeat /= ifeat.norm(dim=-1, keepdim=True)
                probs = (100 * ifeat @ tfeat.T).softmax(dim=-1)[0]
            target_p = probs[0].item()
            nsfw_p = probs[-n_nsfw:].sum().item()
            best_is_distractor = probs.argmax().item() != 0
            if nsfw_p > 0.15:
                nsfw_hits += 1
                os.remove(path)
                removed += 1
            elif target_p < threshold or best_is_distractor:
                os.remove(path)
                removed += 1
        removed_total += removed
        nsfw_total += nsfw_hits
        if nsfw_hits:
            tqdm.write(f"#{dex:>3} {name:<14} dropped {nsfw_hits} NSFW-flagged")
        bar.set_postfix_str(f"{name} (-{removed_total}, {nsfw_total} NSFW)")
    print(f"clip filter done, removed {removed_total} images "
          f"({nsfw_total} NSFW-flagged)")


# --------------------------------------------------------------------------- #
# Stage 0: purge web-scraped images (keep PokeAPI seeds + TCG cards)
# --------------------------------------------------------------------------- #
def stage_purge(classes):
    """Remove search-engine-scraped images (bing/ddg/numeric filenames), which
    were dropped from the pipeline for pulling junk. Keeps seed_* and card_*."""
    removed = 0
    for _, name, _ in tqdm(classes, desc="purge ", unit="cls"):
        d = class_dir(name)
        for fn in os.listdir(d):
            if fn.startswith("seed_") or fn.startswith("card_"):
                continue
            os.remove(os.path.join(d, fn))
            removed += 1
    print(f"purge done, removed {removed} web-scraped images (seeds + cards kept)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="PokeGuess dataset-building agent")
    ap.add_argument("stage",
                    choices=["seed", "cards", "scrape", "dedup", "clip", "purge", "all"])
    ap.add_argument("--min-id", type=int, default=1)
    ap.add_argument("--max-id", type=int, default=251)
    ap.add_argument("--only", nargs="*", help="Limit to specific Pokemon names.")
    ap.add_argument("--per-class", type=int, default=120, help="Target scraped images/class.")
    ap.add_argument("--engines", nargs="+", choices=["bing", "ddg"],
                    default=["bing", "ddg"], help="Image sources to scrape from.")
    ap.add_argument("--threshold", type=float, default=0.22, help="CLIP keep threshold.")
    ap.add_argument("--card-cap", type=int, default=0,
                    help="Max TCG cards/class (0 = unlimited, get them all).")
    ap.add_argument("--crop-frac", type=float, default=0.52,
                    help="Fraction of card height to keep from the top (the artwork).")
    ap.add_argument("--refresh-cards", action="store_true",
                    help="Re-scan the card catalogue instead of using the cache.")
    args = ap.parse_args()

    classes = load_classes(args.min_id, args.max_id, args.only)
    if not classes:
        sys.exit("No classes selected (check --min-id/--max-id/--only).")

    # NOTE: 'all' = seed + cards + dedup + clip. Search-engine scraping (the
    # 'scrape' stage) was removed from the default flow -- it pulled too much
    # junk. Run it explicitly if ever needed.
    if args.stage == "purge":
        stage_purge(classes)
    if args.stage in ("seed", "all"):
        stage_seed(classes)
    if args.stage in ("cards", "all"):
        stage_cards(classes, crop_frac=args.crop_frac, cap=args.card_cap,
                    refresh=args.refresh_cards)
    if args.stage == "scrape":  # opt-in only, not part of 'all'
        stage_scrape(classes, per_class=args.per_class, engines=args.engines)
    if args.stage in ("dedup", "all"):
        stage_dedup(classes)
    if args.stage in ("clip", "all"):
        stage_clip(classes, threshold=args.threshold)


if __name__ == "__main__":
    main()
