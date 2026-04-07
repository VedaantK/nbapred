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


def _parse_market_item(item: dict, target_date: str) -> dict | None:
    """Parse a single Kalshi market/event item into our standard format. Returns None if unparseable."""
    title = item.get("title", item.get("event_title", ""))
    ticker = item.get("ticker", item.get("event_ticker", ""))

    player, line = parse_player_and_line(title)
    if player is None:
        return None

    # Kalshi prices are in cents (0–100) → divide by 100 for probability
    yes_ask = item.get("yes_ask", item.get("last_price", 50)) or 50
    yes_bid = item.get("yes_bid", yes_ask) or yes_ask
    # Guard: if values look like they're already 0-1 decimals, scale up first
    if yes_ask <= 1.0:
        yes_ask = yes_ask * 100
        yes_bid = yes_bid * 100
    mid_prob = round(((yes_ask + yes_bid) / 2) / 100, 4)

    return {
        "source": "kalshi",
        "market_id": ticker,
        "market_title": title,
        "player_name": player,
        "line": line,
        "over_prob": mid_prob,
        "under_prob": round(1 - mid_prob, 4),
        "game_date": target_date,
    }


def search_nba_player_markets(game_date: str | None = None) -> list[dict]:
    """
    Search Kalshi for open NBA player points markets.
    Primary: series-based search for known NBA player scoring series.
    Fallback: generic keyword scan of all open markets.
    """
    from datetime import date as _date
    target_date = game_date or _date.today().strftime("%Y-%m-%d")
    seen_tickers: set[str] = set()
    results = []

    # ── Primary: NBA series tickers for player points ─────────────────────────
    nba_series = ["NBAPTS", "NBA", "NBAPLAYER", "NBAPLAY"]
    for series in nba_series:
        items = _fetch_kalshi_markets_page(
            f"{KALSHI_BASE_URL}/events",
            {"series_ticker": series, "status": "open", "limit": 200},
        )
        for item in items:
            ticker = item.get("ticker", item.get("event_ticker", ""))
            if ticker in seen_tickers:
                continue
            parsed = _parse_market_item(item, target_date)
            if parsed:
                seen_tickers.add(ticker)
                results.append(parsed)

        # Also try /markets with series filter
        items = _fetch_kalshi_markets_page(
            f"{KALSHI_BASE_URL}/markets",
            {"series_ticker": series, "status": "open", "limit": 200},
        )
        for item in items:
            ticker = item.get("ticker", "")
            if ticker in seen_tickers:
                continue
            parsed = _parse_market_item(item, target_date)
            if parsed:
                seen_tickers.add(ticker)
                results.append(parsed)

    # ── Fallback: generic keyword scan if series search found nothing ─────────
    if not results:
        logger.info("Series search found no markets, trying generic keyword scan")
        for endpoint, params in [
            (f"{KALSHI_BASE_URL}/markets", {"status": "open", "limit": 200}),
            (f"{KALSHI_BASE_URL}/events", {"status": "open", "limit": 100}),
        ]:
            items = _fetch_kalshi_markets_page(endpoint, params)
            for item in items:
                title = item.get("title", item.get("event_title", ""))
                ticker = item.get("ticker", item.get("event_ticker", ""))
                if ticker in seen_tickers:
                    continue
                if not any(kw in title.lower() for kw in ["points", "pts", "score"]):
                    continue
                if "nba" not in title.lower() and "nba" not in ticker.lower():
                    if "nba" not in str(item.get("category", "")).lower():
                        continue
                parsed = _parse_market_item(item, target_date)
                if parsed:
                    seen_tickers.add(ticker)
                    results.append(parsed)

    logger.info(f"Found {len(results)} Kalshi NBA player markets for {target_date}")
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
