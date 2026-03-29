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


def search_nba_player_markets(game_date: str | None = None) -> list[dict]:
    """Search Kalshi for open NBA player points markets."""
    from datetime import date as _date
    target_date = game_date or _date.today().strftime("%Y-%m-%d")
    results = []

    # Try fetching events with NBA series
    for endpoint, params in [
        (f"{KALSHI_BASE_URL}/markets", {"status": "open", "limit": 200}),
        (f"{KALSHI_BASE_URL}/events", {"status": "open", "limit": 100}),
    ]:
        try:
            resp = requests.get(endpoint, params=params, timeout=(5, 8))
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning(f"Kalshi API request failed ({endpoint}): {e}")
            continue

        # Markets endpoint returns {"markets": [...]}
        items = data.get("markets", data.get("events", []))
        for item in items:
            title = item.get("title", item.get("event_title", ""))
            ticker = item.get("ticker", item.get("event_ticker", ""))

            # Filter for NBA player points markets
            if not any(kw in title.lower() for kw in ["points", "pts", "score"]):
                continue
            if "nba" not in title.lower() and "nba" not in ticker.lower():
                # Also try category/series field
                if "nba" not in str(item.get("category", "")).lower():
                    continue

            player, line = parse_player_and_line(title)
            if player is None:
                continue

            # Kalshi prices are in cents (0–100), representing probability %
            yes_ask = item.get("yes_ask", item.get("last_price", 50))
            yes_bid = item.get("yes_bid", yes_ask)
            mid_prob = ((yes_ask + yes_bid) / 2) / 100 if yes_ask and yes_bid else None

            results.append({
                "source": "kalshi",
                "market_id": ticker,
                "market_title": title,
                "player_name": player,
                "line": line,
                "over_prob": mid_prob,
                "under_prob": (1 - mid_prob) if mid_prob is not None else None,
                "game_date": target_date,
            })

    logger.info(f"Found {len(results)} Kalshi NBA player markets")
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
