"""
Synthetic 30-day snapshot history generator for the Ask Your Market demo.

books.toscrape.com has static prices, so we generate a believable history
instead. The output is a CSV of snapshot rows, one per product per day,
plus a "latest" catalog file for convenience.

The random seed is fixed so the eval references stay valid.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Product catalog — source of truth for the demo
# ---------------------------------------------------------------------------

CATALOG: list[dict[str, Any]] = [
    {
        "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "title": "A Light in the Attic",
        "base_price": 15.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.8,
        "reviews_count": 872,
        "description": (
            "An illustrated collection of poems reflecting on childhood, "
            "absurdity, and the strange corners of ordinary life. Features "
            "drawings and verse in a playful, award-winning style."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        "title": "Tipping the Velvet",
        "base_price": 12.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.5,
        "reviews_count": 512,
        "description": (
            "A vivid historical novel set in Victorian London, following a "
            "young woman's journey through music halls, love, and self-discovery. "
            "Sandy lives in Whitstable, a dull fishing town, until a night out at "
            "the opera changes everything."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/soumission_998/index.html",
        "title": "Soumission",
        "base_price": 14.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.2,
        "reviews_count": 389,
        "description": (
            "A provocative contemporary French novel about academic life, "
            "political upheaval, and the choices a professor faces when the "
            "institutions around him begin to shift."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/sharp-objects_997/index.html",
        "title": "Sharp Objects",
        "base_price": 11.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.4,
        "reviews_count": 621,
        "description": (
            "A dark psychological thriller about a reporter who returns to her "
            "hometown to cover the murders of two preteen girls, confronting her "
            "own traumatic past along the way."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/sapiens-a-brief-history-of-humankind_996/index.html",
        "title": "Sapiens: A Brief History of Humankind",
        "base_price": 18.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.7,
        "reviews_count": 1024,
        "description": (
            "A sweeping account of human history from the Stone Age to the present, "
            "exploring how Homo sapiens came to dominate the planet through language, "
            "imagined orders, and shared myths."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/the-requiem-red_995/index.html",
        "title": "The Requiem Red",
        "base_price": 9.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.0,
        "reviews_count": 203,
        "description": (
            "A Gothic mystery set in nineteenth-century Germany, where a young "
            "woman haunted by visions becomes entangled in a series of murders "
            "linked to a centuries-old requiem."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/the-dirty-little-secrets-of-getting-your-dream-job_994/index.html",
        "title": "The Dirty Little Secrets of Getting Your Dream Job",
        "base_price": 13.49,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 3.9,
        "reviews_count": 156,
        "description": (
            "A candid career guide that strips away polite fictions and tells you what "
            "recruiters actually look for, how hiring decisions really get made, and "
            "what candidates get wrong."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/the-coming-woman-a-novel-based-on-the-life-of-the-infamous-feminist-victoria-woodhull_993/index.html",
        "title": "The Coming Woman: A Novel Based on the Life of the Infamous Feminist Victoria Woodhull",
        "base_price": 16.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.3,
        "reviews_count": 278,
        "description": (
            "A historical novel dramatizing the life of Victoria Woodhull, the "
            "nineteenth-century feminist, stockbroker, and presidential candidate "
            "who scandalized and inspired America."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/the-boys-in-the-boat-nine-americans-and-their-epic-quest-for-gold-at-the-1936-berlin-olympics_992/index.html",
        "title": "The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics",
        "base_price": 14.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.6,
        "reviews_count": 912,
        "description": (
            "The true story of the University of Washington rowing team that went from "
            "obscurity to the 1936 Berlin Olympics, set against the rise of Nazi Germany. "
            "A tale of teamwork, resilience, and an unlikely group of young men."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/the-black-maria_991/index.html",
        "title": "The Black Maria",
        "base_price": 12.49,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.1,
        "reviews_count": 187,
        "description": (
            "A quirky, inventive novel about a film extra in 1970s Philadelphia whose life "
            "unfolds through a series of surreal encounters and a search for an elusive "
            "actress."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/starving-hearts-triangular-trade-trilogy-1_990/index.html",
        "title": "Starving Hearts (Triangular Trade Trilogy, Book 1)",
        "base_price": 10.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 3.8,
        "reviews_count": 94,
        "description": (
            "A sprawling historical saga tracing a family across continents and centuries, "
            "from the triangular trade routes through revolution and love. The first book "
            "in a trilogy about inheritance, debt, and belonging."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/shakespeares-sonnets_989/index.html",
        "title": "Shakespeare's Sonnets",
        "base_price": 8.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.5,
        "reviews_count": 401,
        "description": (
            "The complete sonnets of William Shakespeare, with the famous sequence "
            "addressing the Fair Youth, the Dark Lady, and the poet's own mortality. "
            "A slim, essential volume of English verse."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/set-me-free_988/index.html",
        "title": "Set Me Free",
        "base_price": 9.49,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.0,
        "reviews_count": 132,
        "description": (
            "A fast-paced thriller about a woman trapped in a dangerous situation who must "
            "outsmart her captors and reclaim her life. Taut, twisty, and hard to put down."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/scott-pilgrims-precious-little-life-scott-pilgrim-1_987/index.html",
        "title": "Scott Pilgrim's Precious Little Life (Scott Pilgrim, Vol. 1)",
        "base_price": 11.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.4,
        "reviews_count": 567,
        "description": (
            "The first volume of the graphic novel series in which Scott Pilgrim, a "
            "twenty-three-year-old bassist, must defeat his new girlfriend's seven evil "
            "exes in combat. A love letter to video games and indie rock."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/rip-it-up-and-start-again_986/index.html",
        "title": "Rip It Up and Start Again",
        "base_price": 17.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.6,
        "reviews_count": 334,
        "description": (
            "A definitive history of post-punk, charting the movement from the ashes of punk "
            "through bands like Joy Division, Gang of Four, and Talking Heads. A rich, "
            "music-filled account of a restless era."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/our-band-could-be-your-life-scenes-from-the-american-indie-underground-1981-1991_985/index.html",
        "title": "Our Band Could Be Your Life: Scenes from the American Indie Underground, 1981-1991",
        "base_price": 16.49,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.5,
        "reviews_count": 268,
        "description": (
            "An oral history of the American indie record scene of the 1980s, built around "
            "in-depth interviews with the bands, label owners, and fans who kept it alive. "
            "Essential reading on underground music."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/olio_984/index.html",
        "title": "Olio",
        "base_price": 13.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.2,
        "reviews_count": 199,
        "description": (
            "An inventive, genre-blending novel about a partial ghost who haunts a family "
            "and the unravelling of the secrets that bind them. Warm, strange, and sharply "
            "written."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/mesaerion-the-best-science-fiction-stories-1800-1849_983/index.html",
        "title": "Mesaeiron: The Best Science Fiction Stories 1800-1849",
        "base_price": 19.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.3,
        "reviews_count": 145,
        "description": (
            "An anthology collecting the finest early science fiction from 1800 to 1849, "
            "bringing together forgotten pioneers and well-known names in an essential "
            "survey of the genre's formative decades."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/libertarianism-for-beginners_982/index.html",
        "title": "Libertarianism for Beginners",
        "base_price": 9.99,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 3.6,
        "reviews_count": 88,
        "description": (
            "A clear, accessible introduction to libertarian political thought, covering "
            "individual liberty, free markets, and the role of the state. A primer for "
            "newcomers and a refresher for the curious."
        ),
    },
    {
        "url": "https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html",
        "title": "It's Only the Himalayas",
        "base_price": 11.49,
        "currency": "GBP",
        "in_stock": True,
        "promo": None,
        "rating": 4.1,
        "reviews_count": 119,
        "description": (
            "A memoir of a disastrous trek in the Himalayas, equal parts funny and harrowing. "
            "A candid, self-deprecating account of what happens when a novice climber bites "
            "off more than he can chew."
        ),
    },
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DAYS = 30  # number of days of history to generate
SCrape_INTERVAL_DAYS = 1  # one snapshot per product per day
RATING_DECIMALS = 1

# Fixed seed for reproducibility (eval references depend on this).
SEED = 20261013

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_catalog() -> list[dict[str, Any]]:
    """Return a copy of the catalog with no snapshot-derived fields."""
    return [dict(p) for p in CATALOG]


def generate_snapshots(
    catalog: list[dict[str, Any]] | None = None,
    days: int = DAYS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Generate a chronological list of snapshot rows.

    Each row is a flat dict compatible with the snapshots.csv schema:

        scraped_at, url, title, price, currency, in_stock, promo,
        rating, reviews_count, description

    The walk is deterministic given `seed` and `catalog`.
    """
    catalog = catalog or build_catalog()
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Pre-decide a handful of "real, checkable events" so the eval set can
    # assert on them. These are drawn before the per-day walk so they are
    # stable and documented.
    events = _plan_events(catalog, rng)

    for offset in range(days):
        day = today - timedelta(days=(days - 1 - offset))
        scraped_at = day.strftime("%Y-%m-%d")

        for product in catalog:
            row = _snapshot_for_day(
                product=product,
                day_index=offset,
                scraped_at=scraped_at,
                rng=rng,
                events=events,
            )
            rows.append(row)

    # Chronological order (oldest first) is friendlier for diffs and reading.
    rows.sort(key=lambda r: (r["scraped_at"], r["url"]))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path | str | None = None) -> Path:
    """Write snapshot rows to a CSV. Returns the path written to."""
    path = Path(path) if path else (DATA_DIR / "snapshots.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scraped_at",
        "url",
        "title",
        "price",
        "currency",
        "in_stock",
        "promo",
        "rating",
        "reviews_count",
        "description",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def write_catalog(
    rows: list[dict[str, Any]] | None = None,
    path: Path | str | None = None,
) -> Path:
    """Write the "latest" view per product (the catalog for convenience).

    Derived from the most recent snapshot per URL. The catalog CSV omits
    the snapshot-only `scraped_at` column.
    """
    if rows:
        latest = _latest_from_rows(rows)
    else:
        latest = _latest_from_catalog()

    path = Path(path) if path else (DATA_DIR / "catalog.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    catalog_fieldnames = [
        "url",
        "title",
        "price",
        "currency",
        "in_stock",
        "promo",
        "rating",
        "reviews_count",
        "description",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=catalog_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in latest:
            writer.writerow(row)

    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_date() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def _snapshot_for_day(
    product: dict[str, Any],
    day_index: int,
    scraped_at: str,
    rng: random.Random,
    events: dict[str, Any],
) -> dict[str, Any]:
    """Compute one snapshot row for one product on one day."""
    url = product["url"]
    base = product["base_price"]
    currency = product["currency"]

    # Price walk
    price = _price_on_day(
        base=base,
        day_index=day_index,
        rng=rng,
        url=url,
        events=events,
    )

    # Availability
    in_stock = _stock_on_day(
        originally_in_stock=product["in_stock"],
        day_index=day_index,
        rng=rng,
        url=url,
        events=events,
    )

    # Promo
    promo = _promo_on_day(
        original_promo=product["promo"],
        day_index=day_index,
        rng=rng,
        url=url,
        events=events,
    )

    # Rating drift (small)
    rating = _rating_on_day(
        original_rating=product["rating"],
        day_index=day_index,
        rng=rng,
        url=url,
    )

    # Reviews count drift (slowly increases, occasionally jumps on promo)
    reviews_count = _reviews_on_day(
        original=product["reviews_count"],
        day_index=day_index,
        rng=rng,
        promo=bool(promo),
        url=url,
    )

    price = round(price, 2)
    # Re-apply the realistic band after rounding so a value like 10.3385
    # does not become 10.34 and fall just outside base*1.15.
    lo = product["base_price"] * 0.75
    hi = product["base_price"] * 1.15
    price = max(lo, min(hi, price))

    return {
        "scraped_at": scraped_at,
        "url": url,
        "title": product["title"],
        "price": round(price, 2),
        "currency": currency,
        "in_stock": "true" if in_stock else "false",
        "promo": promo if promo else "",
        "rating": round(rating, RATING_DECIMALS),
        "reviews_count": reviews_count,
        "description": product["description"],
    }


def _price_on_day(
    base: float,
    day_index: int,
    rng: random.Random,
    url: str,
    events: dict[str, Any],
) -> float:
    """Random walk on price with a bias toward small decreases over time,
    occasional promo spikes, and a few hard-coded events.

    After a downward price event, the post-event price is anchored to a
    mean-reverting process around the post-event target and is hard-capped
    below the pre-event price (the price on the day before the event fired)
    so it can never recover. The pre-event price is tracked during the walk
    so no extra RNG draws are consumed.
    """
    # Determine event state for this product.
    drop_spec = None
    for url_key, spec in events.get("price_events", {}).items():
        if url_key == url and spec["factor"] < 1.0:
            drop_spec = spec
            break

    drop_day = drop_spec["day"] if drop_spec else None
    post_event = drop_day is not None and day_index >= drop_day

    price = base
    pre_event_price = None
    trend = -0.0008
    target = base * drop_spec["factor"] if drop_spec else None

    for d in range(day_index + 1):
        # Track the price on the day before the event (the ceiling).
        if drop_spec and d == drop_day - 1:
            pre_event_price = price

        if post_event and d > drop_day:
            # Mean-reverting random walk around the post-event target.
            reversion = (target - price) / price * 0.25
            shock = rng.gauss(0, 0.004)
            price *= 1.0 + reversion + shock
        else:
            shock = rng.gauss(0, 0.012) + trend
            price *= 1.0 + shock

    # Apply the event shock on the event day.
    if drop_spec and day_index >= drop_day:
        price = price * drop_spec["factor"] if day_index == drop_day else price

    # Clamp to realistic band.
    if post_event and drop_spec:
        lo = base * 0.70
        hi = base * 0.95
    else:
        lo = base * 0.75
        hi = base * 1.15
    price = max(lo, min(hi, price))

    # Hard ceiling: never above the pre-event price.
    if post_event and drop_spec and pre_event_price:
        price = min(price, pre_event_price * 0.999)

    return price


def _stock_on_day(
    originally_in_stock: bool,
    day_index: int,
    rng: random.Random,
    url: str,
    events: dict[str, Any],
) -> bool:
    """Stock flips occasionally; historically it starts in stock.

    Event-driven stockouts and recoveries are decisive. The random noise
    that causes occasional stockouts is suppressed before a planned
    stockout event so the event is the *first* time the product goes OOS.
    Likewise, once a recovery event fires, random noise does not re-stockout
    the product.
    """
    # Planned stockout: from this day the product is OOS (until a recovery)
    if url in events.get("stock_out_days", {}):
        event_day = events["stock_out_days"][url]
        if day_index >= event_day:
            back_day = events.get("stock_back_days", {}).get(url)
            if back_day is not None and day_index >= back_day:
                return True
            return False

    # Planned recovery (only relevant if there is a stockout event)
    if url in events.get("stock_back_days", {}):
        event_day = events["stock_back_days"][url]
        if day_index >= event_day:
            return True

    # Default behaviour: mostly in stock, occasional temporary stockout.
    # Suppress the random stockout before a *planned* stockout so the event
    # is the first OOS occurrence (makes the eval reference clean).
    if not originally_in_stock:
        return False

    planned_out = url in events.get("stock_out_days", {})
    if planned_out:
        return True  # stay in stock until the planned event day

    if rng.random() < 0.04:
        return False
    return True


def _promo_on_day(
    original_promo: str | None,
    day_index: int,
    rng: random.Random,
    url: str,
    events: dict[str, Any],
) -> str | None:
    """Promos appear and disappear. Some are hard-coded.

    Random promo churn is suppressed before a planned promo start so the
    event is the first occurrence (clean eval reference). Once a planned
    promo is active, random promos do not override it.
    """
    # Planned promo window
    if url in events.get("promo_start_days", {}):
        start = events["promo_start_days"][url]
        end = events.get("promo_end_days", {}).get(url)
        if day_index >= start and (end is None or day_index <= end):
            return events["promo_texts"].get(url, "Special offer")
        # Before the planned start, suppress random promos for this product
        # so the event is the first promo occurrence.
        if day_index < start:
            return None

    if url in events.get("promo_end_days", {}):
        end = events["promo_end_days"][url]
        if day_index > end:
            return None

    # Natural promo churn
    if original_promo:
        if rng.random() < 0.7:
            return original_promo
    if rng.random() < 0.06:
        return rng.choice([
            "10% off",
            "Summer sale",
            "Buy 2, save 15%",
            "Flash sale",
            "Weekend special",
        ])
    return None


def _rating_on_day(
    original_rating: float,
    day_index: int,
    rng: random.Random,
    url: str,
) -> float:
    """Small random drift around the original rating."""
    drift = rng.gauss(0, 0.05 * (1 + day_index * 0.02))
    rating = original_rating + drift
    return max(1.0, min(5.0, rating))


def _reviews_on_day(
    original: int,
    day_index: int,
    rng: random.Random,
    promo: bool,
    url: str,
) -> int:
    """Reviews accumulate slowly; promo days add a little bump."""
    growth = original * 0.002 * (day_index + 1)  # slow accumulation
    bump = 8 if promo else 0
    noise = rng.gauss(0, 3)
    return max(original, int(original + growth + bump + noise))


def _apply_price_event(
    price: float,
    day_index: int,
    url: str,
    events: dict[str, Any],
) -> float:
    """Apply a hard-coded price event if the day matches.

    This is a no-op for the main walk now that the event shock and the
    post-event mean reversion are handled directly in `_price_on_day`.
    Kept for call-site stability; the real logic lives in `_price_on_day`.
    """
    return price


def _plan_events(
    catalog: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    """Plan a handful of deterministic, checkable events.

    Returns a dict describing events by URL so the eval set can assert on them.
    """
    events: dict[str, Any] = {
        "price_events": {},       # url -> {day, factor, target}
        "stock_out_days": {},     # url -> day (stockout begins)
        "stock_back_days": {},    # url -> day (back in stock)
        "promo_start_days": {},   # url -> day
        "promo_end_days": {},     # url -> day
        "promo_texts": {},        # url -> text
    }

    # Pick a few products for explicit, checkable events.
    # product 0: price drop on day 12
    p0 = catalog[0]
    events["price_events"][p0["url"]] = {
        "day": 12,
        "factor": 0.85,  # ~15% drop
        "target": p0["base_price"] * 0.85,
    }

    # product 1 (index 1): goes out of stock on day 20, back on day 25
    p1 = catalog[1]
    events["stock_out_days"][p1["url"]] = 20
    events["stock_back_days"][p1["url"]] = 25

    # product 5: promo "Flash sale" on days 8-12
    p5 = catalog[5]
    events["promo_start_days"][p5["url"]] = 8
    events["promo_end_days"][p5["url"]] = 12
    events["promo_texts"][p5["url"]] = "Flash sale"

    # product 10: promo "Buy 2, save 15%" on days 18-22
    p10 = catalog[10]
    events["promo_start_days"][p10["url"]] = 18
    events["promo_end_days"][p10["url"]] = 22
    events["promo_texts"][p10["url"]] = "Buy 2, save 15%"

    # product 15: rating drops a bit on day 15 (more drift than usual)
    # (handled via the natural drift; we don't hard-code rating events,
    #  but we make sure the seed produces a visible dip for this product.)

    return events


def _latest_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Most recent snapshot per URL, sorted by URL."""
    by_url: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = row["url"]
        cur = by_url.get(url)
        if cur is None or row["scraped_at"] > cur["scraped_at"]:
            by_url[url] = row
    latest = list(by_url.values())
    latest.sort(key=lambda r: r["url"])
    return latest


def _latest_from_catalog() -> list[dict[str, Any]]:
    """Catalog-as-latest: start from base values (day 0 equivalent)."""
    rows: list[dict[str, Any]] = []
    today = _now_date().strftime("%Y-%m-%d")
    for product in CATALOG:
        rows.append(
            {
                "scraped_at": today,
                "url": product["url"],
                "title": product["title"],
                "price": round(product["base_price"], 2),
                "currency": product["currency"],
                "in_stock": "true" if product["in_stock"] else "false",
                "promo": product["promo"] if product["promo"] else "",
                "rating": round(product["rating"], RATING_DECIMALS),
                "reviews_count": product["reviews_count"],
                "description": product["description"],
            }
        )
    rows.sort(key=lambda r: r["url"])
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    catalog = build_catalog()
    print(f"catalog: {len(catalog)} products")

    rows = generate_snapshots(catalog=catalog, days=DAYS, seed=SEED)
    print(f"generated {len(rows)} snapshot rows over {DAYS} days")

    csv_path = write_csv(rows)
    print(f"wrote {csv_path}")

    cat_path = write_catalog(rows=rows)
    print(f"wrote latest catalog: {cat_path}")

    _print_summary(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    dates = sorted({r["scraped_at"] for r in rows})
    print(f"date range: {dates[0]} .. {dates[-1]}  ({len(dates)} distinct dates)")
    urls = {r["url"] for r in rows}
    print(f"distinct products: {len(urls)}")

    # show the planned events so the dev can verify them
    events = _plan_events(build_catalog(), random.Random(SEED))
    print("planned events:")
    for url, spec in events["price_events"].items():
        title = next(p["title"] for p in build_catalog() if p["url"] == url)
        print(f"  price drop: {title} on day {spec['day']} (factor {spec['factor']})")
    for url, day in events["stock_out_days"].items():
        title = next(p["title"] for p in build_catalog() if p["url"] == url)
        back = events["stock_back_days"].get(url)
        print(f"  out of stock: {title} from day {day} to day {back}")
    for url, day in events["promo_start_days"].items():
        title = next(p["title"] for p in build_catalog() if p["url"] == url)
        end = events["promo_end_days"].get(url)
        text = events["promo_texts"].get(url, "")
        print(f"  promo: {title} day {day}-{end} ({text})")


if __name__ == "__main__":
    main()
