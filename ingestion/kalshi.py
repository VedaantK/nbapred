"""
NBA player points markets from Kalshi prediction market.
No API key required for public market data.
"""

import re
import sqlite3
from datetime import date
from pathlib import Path

import requests

from config.settings import KALSHI_BASE_URL, DB_PATH
from config.logging_config import setup_logging
from utils.name_matcher import normalize_name

logger = setup_logging("kalshi")

# Regex patterns to extract player name and points line from Kalshi market titles
_SCORE_PATTERNS = [
    # "Will LeBron James score 25+ points?"
    re.compile(r"will\s+(.+?)\s+score\s+(\d+\.?\d*)\+?\s*points?", re.IGNORECASE),
    # "LeBron James to score over 25.5 points"
    re.compile(r"(.+?)\s+to score (?:over|more than)\s+(\d+\.?\d*)\s*points?", re.IGNORECASE),
    # "LeBron James 25+ Points"
    re.compile(r"(.+?)\s+(\d+\.?\d*)\+\s*points?", re.IGNORECASE),
    # "NBA: LeBron James Over 25.5 Pts"
    re.compile(r"(?:nba[:\s]+)?(.+?)\s+over\s+(\d+\.?\d*)\s+pts?", re.IGNORECASE),
]


def parse_player_and_line(market_title: str) -> tuple[str | None, float | None]:
    """
    Extract player name and points line from a Kalshi market title.
    Returns (player_name, points_line) or (None, None) if not parseable.
    """
    for pattern in _SCORE_PATTERNS:
        m = pattern.search(market_title)
        if m:
            player = m.group(1).strip()
            try:
                line = float(m.group(2))
                return player, line
            except ValueError:
                continue
    return None, None


def _fetch_kalshi_markets_page(endpoint: str, params: dict) -> list[dict]:
    """Fetch one page from a Kalshi endpoint with pagination via cursor."""
    all_items = []
    cursor = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        try:
            resp = requests.get(endpoint, params=p, timeout=(5, 15))
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning(f"Kalshi API request failed ({endpoint}): {e}")
            break
        items = data.get("markets", data.get("events", []))
        all_items.extend(items)
        cursor = data.get("cursor")
        if not cursor or not items:
            break
    return all_items


# Kalshi's NBA player-points series. Each player gets a ladder of markets
# ("Keldon Johnson: 15+ points", "...: 20+ points"), one per strike.
PLAYER_POINTS_SERIES = "KXNBAPTS"
_EVENT_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")  # KXNBAPTS-26NOV09SASPHX -> 26 NOV 09
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _event_date(event_ticker: str) -> str | None:
    m = _EVENT_DATE.search(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    yy, mon, dd = m.groups()
    return f"20{yy}-{_MONTHS[mon]:02d}-{dd}"


def _dollars(item: dict, key: str) -> float | None:
    """Kalshi now quotes prices as dollar strings ('0.0800'); 0 means no quote."""
    try:
        value = float(item.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_market_item(item: dict, target_date: str) -> dict | None:
    """Parse one KXNBAPTS market. Returns None if it's for another date or has no quote."""
    if _event_date(item.get("event_ticker", "")) != target_date:
        return None
    title = item.get("title", "")
    player = title.split(":")[0].strip() if ":" in title else None
    line = item.get("floor_strike")  # "20+ points" has floor_strike 19.5, i.e. over 19.5
    if not player or line is None or item.get("strike_type") != "greater":
        return None

    bid, ask = _dollars(item, "yes_bid_dollars"), _dollars(item, "yes_ask_dollars")
    if bid is None and ask is None:
        return None  # nobody quoting: no price information at all
    mid = (bid + ask) / 2 if bid is not None and ask is not None else (bid or ask)
    mid = round(mid, 4)

    return {
        "source": "kalshi",
        "market_id": item.get("ticker", ""),
        "market_title": title,
        "player_name": player,
        "line": float(line),
        "over_prob": mid,
        "under_prob": round(1 - mid, 4),
        "game_date": target_date,
    }


def search_nba_player_markets(game_date: str | None = None) -> list[dict]:
    """
    Open Kalshi NBA player-points markets for one date, ONE row per player.

    From each player's ladder keep the strike priced closest to 50%, which is
    the equivalent of a sportsbook's main line. Downstream code (predict.py)
    assumes one line per player per date, so a whole ladder would mix a line
    from one strike with a probability from another.
    """
    from datetime import date as _date
    target_date = game_date or _date.today().strftime("%Y-%m-%d")

    items = _fetch_kalshi_markets_page(
        f"{KALSHI_BASE_URL}/markets",
        {"series_ticker": PLAYER_POINTS_SERIES, "status": "open", "limit": 200},
    )
    best: dict[str, dict] = {}
    for item in items:
        parsed = _parse_market_item(item, target_date)
        if not parsed:
            continue
        key = normalize_name(parsed["player_name"])
        cur = best.get(key)
        if cur is None or abs(parsed["over_prob"] - 0.5) < abs(cur["over_prob"] - 0.5):
            best[key] = parsed

    results = list(best.values())
    logger.info(f"Found {len(results)} Kalshi NBA players ({len(items)} open markets) for {target_date}")
    return results


def fetch_and_store_kalshi_markets(db_path: str | Path = DB_PATH, game_date: str | None = None):
    """Pull all open NBA player points markets, parse, and store."""
    db_path = Path(db_path)
    markets = search_nba_player_markets(game_date)

    if not markets:
        logger.info("No Kalshi NBA markets found today")
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    stored = 0

    try:
        for m in markets:
            try:
                conn.execute(
                    """
                    INSERT INTO prediction_market_lines
                    (source, player_name, market_title, market_id, game_date, line, over_prob, under_prob)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        m["source"],
                        m["player_name"],
                        m["market_title"],
                        m["market_id"],
                        m["game_date"],
                        m["line"],
                        m["over_prob"],
                        m["under_prob"],
                    ),
                )
                stored += 1
            except Exception as e:
                logger.debug(f"Failed to store Kalshi market: {e}")

        conn.commit()
    finally:
        conn.close()
    logger.info(f"Stored {stored} Kalshi market records")
