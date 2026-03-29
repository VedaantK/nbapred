"""
NBA player points markets from Polymarket prediction market.
Uses Gamma API for market metadata and CLOB API for live pricing.
No API key required.
"""

import re
import sqlite3
from datetime import date
from pathlib import Path

import requests

from config.settings import POLYMARKET_BASE_URL, POLYMARKET_GAMMA_URL, DB_PATH
from config.logging_config import setup_logging

logger = setup_logging("polymarket")

_SCORE_PATTERNS = [
    re.compile(r"will\s+(.+?)\s+score\s+(\d+\.?\d*)\+?\s*points?", re.IGNORECASE),
    re.compile(r"(.+?)\s+over\s+(\d+\.?\d*)\s+points?", re.IGNORECASE),
    re.compile(r"(.+?)\s+(\d+\.?\d*)\+\s*points?", re.IGNORECASE),
    re.compile(r"(.+?)\s+to score (?:over|more than)\s+(\d+\.?\d*)", re.IGNORECASE),
]


def _parse_player_and_line(question: str) -> tuple[str | None, float | None]:
    for pattern in _SCORE_PATTERNS:
        m = pattern.search(question)
        if m:
            player = m.group(1).strip()
            try:
                line = float(m.group(2))
                return player, line
            except ValueError:
                continue
    return None, None


def search_nba_player_markets(game_date: str | None = None) -> list[dict]:
    """Search Polymarket Gamma API for NBA player scoring markets."""
    from datetime import date as _date
    target_date = game_date or _date.today().strftime("%Y-%m-%d")
    results = []

    for tag in ["NBA", "nba", "basketball"]:
        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/markets",
                params={"tag": tag, "active": "true", "closed": "false", "limit": 200},
                timeout=(5, 8),
            )
            resp.raise_for_status()
            markets = resp.json()
            if isinstance(markets, dict):
                markets = markets.get("markets", [])
        except requests.RequestException as e:
            logger.warning(f"Polymarket Gamma API request failed (tag={tag}): {e}")
            continue

        for m in markets:
            question = m.get("question", "")
            # Filter for player scoring markets
            if not any(kw in question.lower() for kw in ["score", "points", "pts"]):
                continue

            player, line = _parse_player_and_line(question)
            if player is None:
                continue

            # outcomePrices is list of probabilities as strings
            outcome_prices = m.get("outcomePrices", [])
            outcomes = m.get("outcomes", [])

            over_prob = None
            under_prob = None
            if outcome_prices and len(outcome_prices) >= 2:
                try:
                    probs = [float(p) for p in outcome_prices]
                    # Typically ["Yes", "No"] or ["Over", "Under"]
                    for i, outcome in enumerate(outcomes):
                        outcome_lower = str(outcome).lower()
                        if outcome_lower in ("yes", "over") and i < len(probs):
                            over_prob = probs[i]
                        elif outcome_lower in ("no", "under") and i < len(probs):
                            under_prob = probs[i]
                    # Fallback: first price = yes/over
                    if over_prob is None:
                        over_prob = probs[0]
                        under_prob = probs[1] if len(probs) > 1 else (1 - over_prob)
                except (ValueError, IndexError):
                    pass

            # Try CLOB for live price if we have token IDs
            clob_token_ids = m.get("clobTokenIds", [])
            if clob_token_ids and over_prob is None:
                over_prob = get_market_prices(clob_token_ids[0])
                if over_prob is not None:
                    under_prob = 1 - over_prob

            results.append({
                "source": "polymarket",
                "market_id": str(m.get("id", "")),
                "market_title": question,
                "player_name": player,
                "line": line,
                "over_prob": over_prob,
                "under_prob": under_prob,
                "game_date": target_date,
            })

        if results:
            break  # Found markets with this tag, no need to try others

    logger.info(f"Found {len(results)} Polymarket NBA player markets")
    return results


def get_market_prices(token_id: str) -> float | None:
    """Get current mid-market probability from CLOB API."""
    try:
        resp = requests.get(
            f"{POLYMARKET_BASE_URL}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=(5, 8),
        )
        resp.raise_for_status()
        data = resp.json()
        price = data.get("price")
        return float(price) if price is not None else None
    except Exception as e:
        logger.debug(f"CLOB price fetch failed for token {token_id}: {e}")
        return None


def fetch_and_store_polymarket_markets(db_path: str | Path = DB_PATH, game_date: str | None = None):
    """Pull all relevant NBA player points markets, store in DB."""
    db_path = Path(db_path)
    markets = search_nba_player_markets(game_date)

    if not markets:
        logger.info("No Polymarket NBA markets found today — this is common")
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
                logger.debug(f"Failed to store Polymarket market: {e}")

        conn.commit()
    finally:
        conn.close()
    logger.info(f"Stored {stored} Polymarket market records")
