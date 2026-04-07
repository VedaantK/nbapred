"""
Live Kalshi market scanner.
Runs every 15 minutes in the background, fetches current Kalshi prices,
compares against model predictions, and places paper bets when edge is good.
"""

import sqlite3
from datetime import date, datetime
from pathlib import Path

from config.settings import DB_PATH
from config.logging_config import setup_logging
from ingestion.kalshi import search_nba_player_markets
from models.paper_trader import (
    american_to_decimal, kelly_fraction, get_bankroll,
    _get_recent_accuracy,
)
from models.rl_agent import RLBettingAgent, encode_state, ACTION_KELLY_MULT

logger = setup_logging("market_scanner")

SCAN_INTERVAL_SECONDS = 900    # 15 minutes
SCANNER_MIN_EDGE = 0.05         # 5% edge required to place a bet
MAX_KELLY_FRACTION = 0.20


def _time_bucket() -> str:
    """Return the current time-of-day bucket (used as RL state dimension)."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


def fetch_kalshi_snapshot(game_date: str, db_path: str | Path = DB_PATH) -> list[dict]:
    """
    Fetch current Kalshi NBA markets and store them as snapshots.
    Returns the list of markets fetched.
    """
    db_path = Path(db_path)
    markets = search_nba_player_markets(game_date)

    if not markets:
        logger.info(f"No Kalshi markets found for {game_date}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        for m in markets:
            try:
                conn.execute(
                    """
                    INSERT INTO kalshi_price_snapshots
                    (market_id, player_name, line, over_prob, under_prob, game_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.get("market_id", ""),
                        m.get("player_name"),
                        m.get("line"),
                        m.get("over_prob"),
                        m.get("under_prob"),
                        game_date,
                    ),
                )
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Stored {len(markets)} Kalshi snapshots for {game_date}")
    return markets


def find_opportunities(game_date: str, db_path: str | Path = DB_PATH) -> list[dict]:
    """
    Compare the most recent Kalshi snapshot against model predictions.
    Returns ranked list of betting opportunities where edge >= SCANNER_MIN_EDGE.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        # Load averaged predictions for today (skip low-confidence)
        pred_rows = conn.execute(
            """
            SELECT
                player_name,
                AVG(predicted_over_prob) AS model_prob,
                sportsbook_line,
                AVG(edge) AS edge,
                confidence
            FROM predictions
            WHERE game_date = ? AND confidence != 'low'
            GROUP BY player_name
            """,
            (game_date,),
        ).fetchall()

        if not pred_rows:
            return []

        preds_by_name = {}
        for row in pred_rows:
            player_name, model_prob, line, edge, confidence = row
            preds_by_name[player_name.lower()] = {
                "player_name": player_name,
                "model_prob": model_prob,
                "line": line,
                "edge": edge,
                "confidence": confidence,
            }

        # Load the most recent Kalshi snapshot per player for today
        kalshi_rows = conn.execute(
            """
            SELECT player_name, line, over_prob, under_prob
            FROM kalshi_price_snapshots
            WHERE game_date = ?
            AND snapshot_time = (
                SELECT MAX(snapshot_time)
                FROM kalshi_price_snapshots AS k2
                WHERE k2.game_date = kalshi_price_snapshots.game_date
                AND k2.player_name = kalshi_price_snapshots.player_name
            )
            """,
            (game_date,),
        ).fetchall()

        if not kalshi_rows:
            return []

        opportunities = []
        for k_name, k_line, k_over_prob, k_under_prob in kalshi_rows:
            if k_name is None or k_over_prob is None:
                continue

            # Match to prediction (case-insensitive, partial)
            pred = None
            k_lower = k_name.lower()
            for p_lower, p_data in preds_by_name.items():
                if k_lower in p_lower or p_lower in k_lower:
                    pred = p_data
                    break

            if pred is None:
                continue

            model_prob = pred["model_prob"]
            if model_prob is None:
                continue

            # Edge vs Kalshi current price
            over_edge = model_prob - k_over_prob
            under_edge = (1.0 - model_prob) - k_under_prob

            # Pick the direction with greater edge
            if abs(over_edge) >= abs(under_edge) and abs(over_edge) >= SCANNER_MIN_EDGE:
                direction = "over"
                edge = over_edge
                implied_prob = k_over_prob
                dec_odds = 1.0 / max(k_over_prob, 0.01)
            elif abs(under_edge) >= SCANNER_MIN_EDGE:
                direction = "under"
                edge = under_edge
                implied_prob = k_under_prob
                dec_odds = 1.0 / max(k_under_prob, 0.01)
            else:
                continue

            opportunities.append({
                "player_name": k_name,
                "line": k_line or pred["line"],
                "kalshi_over_prob": round(k_over_prob, 4),
                "kalshi_under_prob": round(k_under_prob, 4),
                "model_prob": round(model_prob, 4),
                "edge": round(edge, 4),
                "direction": direction,
                "confidence": pred["confidence"],
                "decimal_odds": round(dec_odds, 4),
                "implied_prob": round(implied_prob, 4),
            })

        opportunities.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return opportunities

    finally:
        conn.close()


def scan_and_place_bets(game_date: str, db_path: str | Path = DB_PATH) -> dict:
    """
    Main scanner entry point:
    1. Fetch latest Kalshi snapshot
    2. Find opportunities vs model predictions
    3. Place bets for new opportunities (skip already-bet players)
    Returns {scanned, new_bets, opportunities}
    """
    db_path = Path(db_path)

    # Fetch fresh snapshot
    markets = fetch_kalshi_snapshot(game_date, db_path)

    # Find opportunities from snapshot vs predictions
    opportunities = find_opportunities(game_date, db_path)

    if not opportunities:
        logger.info(f"Scanner: no opportunities found for {game_date}")
        return {"scanned": len(markets), "new_bets": 0, "opportunities": []}

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    new_bets = []

    try:
        # Load players already bet on today
        existing_bets = {
            row[0].lower()
            for row in conn.execute(
                "SELECT player_name FROM paper_bets WHERE game_date = ?", (game_date,)
            ).fetchall()
        }

        current_balance = get_bankroll(db_path)
        rl_agent = RLBettingAgent.load()
        recent_acc = _get_recent_accuracy(conn)
        time_bucket = _time_bucket()

        for opp in opportunities:
            player_name = opp["player_name"]

            # Skip if already bet on this player today
            if player_name.lower() in existing_bets:
                continue

            # Kelly sizing
            base_frac = kelly_fraction(opp["model_prob"], opp["decimal_odds"])
            if base_frac <= 0:
                continue

            # RL agent decides Kelly multiplier
            rl_state = encode_state(opp["edge"], opp["confidence"], recent_acc, time_bucket)
            rl_action = rl_agent.choose_action(rl_state)
            rl_kelly_mult = ACTION_KELLY_MULT[rl_action]

            if rl_kelly_mult <= 0:
                logger.debug(f"Scanner: RL skipped {player_name} (action=0)")
                continue

            frac = round(base_frac * rl_kelly_mult, 4)
            bet_amount = round(frac * current_balance, 2)
            if bet_amount < 0.01:
                continue

            conn.execute(
                """
                INSERT INTO paper_bets
                (game_date, player_name, bet_direction, bet_amount, odds_source,
                 implied_prob, model_prob, edge, kelly_fraction, decimal_odds, status,
                 rl_state, rl_action, scan_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?,?)
                """,
                (
                    game_date, player_name, opp["direction"], bet_amount, "kalshi",
                    opp["implied_prob"], opp["model_prob"],
                    opp["edge"], frac, opp["decimal_odds"],
                    rl_state, str(rl_action), time_bucket,
                ),
            )

            current_balance = round(current_balance - bet_amount, 2)
            conn.execute(
                """
                INSERT INTO bankroll_history (event_date, action, amount, balance_after, notes)
                VALUES (?, 'bet_placed', ?, ?, ?)
                """,
                (
                    game_date, -bet_amount, current_balance,
                    f"Scanner bet {opp['direction']} on {player_name} via kalshi "
                    f"(edge={opp['edge']:.1%}, {time_bucket})",
                ),
            )

            existing_bets.add(player_name.lower())
            new_bets.append({
                "player_name": player_name,
                "direction": opp["direction"],
                "amount": bet_amount,
                "edge": opp["edge"],
                "kalshi_price": opp["implied_prob"],
                "model_prob": opp["model_prob"],
                "time_bucket": time_bucket,
            })

        if new_bets:
            conn.commit()
            rl_agent.save()
            logger.info(
                f"Scanner placed {len(new_bets)} new bets for {game_date} "
                f"({time_bucket}), bankroll now ${current_balance:.2f}"
            )

    finally:
        conn.close()

    return {
        "scanned": len(markets),
        "new_bets": len(new_bets),
        "opportunities": opportunities,
        "bets_placed": new_bets,
    }
