"""
Sportsbook player prop lines via The Odds API.
Free tier: 500 requests/month — cache aggressively.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from config.settings import ODDS_API_KEY, DB_PATH, CACHE_DIR
from config.logging_config import setup_logging

logger = setup_logging("odds_api")

BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_TTL_HOURS = 2

# (connect_timeout, read_timeout)
_TIMEOUT = (5, 8)


def _cache_path(name: str, game_date: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{game_date}.json"


def _load_cache(name: str, game_date: str) -> dict | None:
    path = _cache_path(name, game_date)
    if not path.exists():
        return None
    today = date.today().strftime("%Y-%m-%d")
    # Historical dates: cache forever. Today: respect TTL.
    if game_date != today:
        with open(path) as f:
            return json.load(f)
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    if age_hours > CACHE_TTL_HOURS:
        return None
    with open(path) as f:
        return json.load(f)


def _save_cache(name: str, game_date: str, data):
    with open(_cache_path(name, game_date), "w") as f:
        json.dump(data, f)


def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None:
        return 0.5
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def get_todays_nba_events(game_date: str | None = None) -> list[dict]:
    """
    Fetch NBA events for the given date from The Odds API.
    Fetches ALL upcoming events then filters client-side by commence_time,
    because the events endpoint does not reliably support commenceTimeFrom/To.
    """
    target_date = game_date or date.today().strftime("%Y-%m-%d")

    cached = _load_cache("nba_events", target_date)
    if cached is not None:
        logger.info(f"Loaded {len(cached)} NBA events from cache ({target_date})")
        return cached

    url = f"{BASE_URL}/sports/basketball_nba/events"
    params = {"apiKey": ODDS_API_KEY, "regions": "us"}

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API requests remaining: {remaining}")
        resp.raise_for_status()
        all_events = resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NBA events: {e}")
        return []

    # Filter to events whose calendar date in US Eastern time matches target_date.
    # NBA games run 12 PM – 11 PM ET. Using UTC-5 (EST) as a safe approximation
    # is accurate for all typical tip-off times regardless of DST.
    target_events = []
    for event in all_events:
        commence = event.get("commence_time", "")
        if not commence:
            continue
        try:
            event_utc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            # Shift to approximate Eastern time (UTC-5); covers EDT games too
            event_et_date = (event_utc - timedelta(hours=5)).strftime("%Y-%m-%d")
        except Exception:
            continue
        if event_et_date == target_date:
            target_events.append(event)

    logger.info(f"Found {len(target_events)} events for {target_date} ({len(all_events)} total upcoming)")
    _save_cache("nba_events", target_date, target_events)
    return target_events


def get_player_points_props(event_id: str, game_date: str) -> list[dict]:
    """For a given event, fetch player points over/under lines."""
    cached = _load_cache(f"props_{event_id}", game_date)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_points",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(f"Fetched props for event {event_id}. Requests remaining: {remaining}")
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch props for event {event_id}: {e}")
        return []

    results = []
    bookmakers = data.get("bookmakers", [])
    for bookmaker in bookmakers:
        bk_name = bookmaker.get("key", "unknown")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_points":
                continue
            outcomes = market.get("outcomes", [])
            player_odds: dict[str, dict] = {}
            for outcome in outcomes:
                player = outcome.get("description", outcome.get("name", ""))
                side = outcome.get("name", "").lower()
                price = outcome.get("price")
                point = outcome.get("point")
                if player not in player_odds:
                    player_odds[player] = {"line": point, "over_price": None, "under_price": None}
                if "over" in side:
                    player_odds[player]["over_price"] = price
                    if point is not None:
                        player_odds[player]["line"] = point
                elif "under" in side:
                    player_odds[player]["under_price"] = price

            for player, odds in player_odds.items():
                over_prob = american_odds_to_prob(odds["over_price"])
                under_prob = american_odds_to_prob(odds["under_price"])
                results.append({
                    "player_name": player,
                    "bookmaker": bk_name,
                    "line": odds["line"],
                    "over_price": odds["over_price"],
                    "under_price": odds["under_price"],
                    "implied_over_prob": round(over_prob, 4),
                    "implied_under_prob": round(under_prob, 4),
                    "game_date": game_date,
                    "market": "player_points",
                })

    _save_cache(f"props_{event_id}", game_date, results)
    return results


def fetch_and_store_all_props(db_path: str | Path = DB_PATH, game_date: str | None = None):
    """Main entry point: fetch all props for the given date, store in DB."""
    db_path = Path(db_path)
    target_date = game_date or date.today().strftime("%Y-%m-%d")

    events = get_todays_nba_events(target_date)
    if not events:
        logger.warning(f"No NBA events found for {target_date}")
        return

    event_ids = [e.get("id", "") for e in events if e.get("id")]
    if not event_ids:
        logger.warning("Events returned but none had IDs")
        return

    logger.info(f"Fetching props for {len(event_ids)} events in parallel ({target_date})...")

    # Do NOT use `with ThreadPoolExecutor` — its __exit__ calls shutdown(wait=True)
    # which blocks until all threads finish, defeating the as_completed timeout.
    all_props: list[dict] = []
    n_workers = max(1, min(len(event_ids), 6))
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = {executor.submit(get_player_points_props, eid, target_date): eid for eid in event_ids}
        try:
            for future in as_completed(futures, timeout=45):
                try:
                    all_props.extend(future.result())
                except Exception as e:
                    logger.warning(f"A props fetch failed: {e}")
        except TimeoutError:
            logger.warning("Timed out waiting for some event props — storing partial results")
    finally:
        executor.shutdown(wait=False)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    total_stored = 0
    try:
        for prop in all_props:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sportsbook_lines
                    (player_name, market, game_date, bookmaker, line, over_price, under_price,
                     implied_over_prob, implied_under_prob)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        prop["player_name"],
                        prop["market"],
                        prop["game_date"],
                        prop["bookmaker"],
                        prop["line"],
                        prop["over_price"],
                        prop["under_price"],
                        prop["implied_over_prob"],
                        prop["implied_under_prob"],
                    ),
                )
                total_stored += 1
            except Exception as e:
                logger.debug(f"Failed to store prop: {e}")
        conn.commit()
    finally:
        conn.close()
    logger.info(f"Stored {total_stored} sportsbook prop lines for {target_date}")
