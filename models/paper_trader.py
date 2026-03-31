"""
Paper prediction market trader.
Uses Kelly criterion to size bets based on model edge vs sportsbook/market odds.
Tracks bankroll, open bets, and P&L over time.
"""

import sqlite3
from datetime import date
from pathlib import Path

from config.settings import DB_PATH
from config.logging_config import setup_logging
from models.rl_agent import RLBettingAgent, encode_state, ACTION_KELLY_MULT
from datetime import datetime as _dt


def _time_bucket() -> str:
    hour = _dt.now().hour
    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"

logger = setup_logging("paper_trader")

DEFAULT_BANKROLL = 30.0
MIN_EDGE = 0.10
KELLY_MULTIPLIER = 0.5   # half-Kelly for safety
MAX_KELLY_FRACTION = 0.20  # never risk more than 20% of bankroll on one bet


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds (-110, +150) to decimal (1.909, 2.5)."""
    if american_odds is None:
        return 1.909  # default -110
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Half-Kelly bet fraction: 0.5 * (b*p - q) / b, capped at MAX_KELLY_FRACTION.
    b = decimal_odds - 1, p = model_prob, q = 1 - p
    Returns 0 if the bet has no edge.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    raw = (b * model_prob - q) / b
    f = KELLY_MULTIPLIER * raw
    return max(0.0, min(f, MAX_KELLY_FRACTION))


def get_bankroll(db_path: str | Path = DB_PATH) -> float:
    """Return current bankroll (latest balance_after from bankroll_history)."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        row = conn.execute(
            "SELECT balance_after FROM bankroll_history ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else DEFAULT_BANKROLL
    finally:
        conn.close()


def reset_bankroll(starting_amount: float, db_path: str | Path = DB_PATH):
    """Reset bankroll to starting_amount, recording a 'reset' event."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        conn.execute(
            """
            INSERT INTO bankroll_history (event_date, action, amount, balance_after, notes)
            VALUES (?, 'reset', ?, ?, 'Manual bankroll reset')
            """,
            (date.today().strftime("%Y-%m-%d"), starting_amount, starting_amount),
        )
        conn.commit()
        logger.info(f"Bankroll reset to ${starting_amount:.2f}")
    finally:
        conn.close()


def _get_recent_accuracy(conn: sqlite3.Connection, window_days: int = 14) -> float | None:
    """Return the model's directional accuracy over the last window_days days."""
    from datetime import timedelta
    cutoff = (date.today() - timedelta(days=window_days)).strftime("%Y-%m-%d")
    row = conn.execute(
        """
        SELECT AVG(CASE WHEN model_was_correct = 1 THEN 1.0 ELSE 0.0 END)
        FROM predictions
        WHERE game_date >= ? AND model_was_correct IS NOT NULL
        """,
        (cutoff,),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def generate_paper_bets(
    game_date: str,
    db_path: str | Path = DB_PATH,
    min_edge: float = MIN_EDGE,
) -> list[dict]:
    """
    Generate paper bet recommendations for game_date based on existing predictions.
    Sizes bets using Kelly criterion against sportsbook, Kalshi, and Polymarket odds.
    Inserts bets into paper_bets and deducts from bankroll_history.
    Returns list of bet dicts.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        # Check if bets already generated for this date
        existing = conn.execute(
            "SELECT COUNT(*) FROM paper_bets WHERE game_date = ?", (game_date,)
        ).fetchone()[0]
        if existing > 0:
            logger.info(f"Bets already generated for {game_date} ({existing} bets)")
            rows = conn.execute(
                "SELECT * FROM paper_bets WHERE game_date = ? ORDER BY ABS(edge) DESC", (game_date,)
            ).fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM paper_bets LIMIT 0").description or []]
            # fallback: return minimal dicts
            return [dict(zip([c[0] for c in conn.execute("PRAGMA table_info(paper_bets)").fetchall()], row)) for row in rows]

        # Fetch deduplicated predictions (avg across models)
        preds = conn.execute(
            """
            SELECT
                player_name,
                AVG(predicted_over_prob) AS model_prob,
                sportsbook_line,
                sportsbook_implied_prob,
                kalshi_prob,
                polymarket_prob,
                AVG(edge) AS edge,
                confidence
            FROM predictions
            WHERE game_date = ?
            GROUP BY player_name
            HAVING confidence = 'high' AND ABS(AVG(edge)) >= ?
            ORDER BY ABS(AVG(edge)) DESC
            """,
            (game_date, min_edge),
        ).fetchall()

        if not preds:
            logger.info(f"No qualifying predictions for {game_date} (min_edge={min_edge:.0%})")
            return []

        current_balance = get_bankroll(db_path)
        generated = []

        for row in preds:
            (player_name, model_prob, sb_line, sb_implied_prob,
             kalshi_prob, polymarket_prob, edge, confidence) = row

            if model_prob is None or edge is None:
                continue

            bet_direction = "over" if edge > 0 else "under"

            # Only bet via Kalshi — skip players without an active Kalshi market
            if kalshi_prob is None:
                continue

            if bet_direction == "over":
                implied = kalshi_prob
                source_edge = model_prob - kalshi_prob
                dec_odds = 1.0 / max(kalshi_prob, 0.01)
            else:
                implied = 1.0 - kalshi_prob
                source_edge = (1.0 - model_prob) - implied
                dec_odds = 1.0 / max(implied, 0.01)

            if source_edge < min_edge:
                continue

            best = {
                "source": "kalshi",
                "decimal_odds": dec_odds,
                "implied_prob": implied,
                "source_edge": source_edge,
            }

            # Base Kelly fraction
            base_frac = kelly_fraction(model_prob, best["decimal_odds"])
            if base_frac <= 0:
                continue

            # RL agent chooses Kelly multiplier based on learned state
            rl_agent = RLBettingAgent.load()
            recent_acc = _get_recent_accuracy(conn)
            rl_state = encode_state(best["source_edge"], confidence, recent_acc, _time_bucket())
            rl_action = rl_agent.choose_action(rl_state)
            rl_kelly_mult = ACTION_KELLY_MULT[rl_action]

            if rl_kelly_mult <= 0:
                continue  # RL agent chose to skip

            frac = round(base_frac * rl_kelly_mult, 4)
            if frac <= 0:
                continue

            bet_amount = round(frac * current_balance, 2)
            if bet_amount < 0.01:
                continue

            # Insert bet with RL metadata
            conn.execute(
                """
                INSERT INTO paper_bets
                (game_date, player_name, bet_direction, bet_amount, odds_source,
                 implied_prob, model_prob, edge, kelly_fraction, decimal_odds, status,
                 rl_state, rl_action)
                VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?)
                """,
                (
                    game_date, player_name, bet_direction, bet_amount,
                    best["source"], best["implied_prob"], model_prob,
                    round(best["source_edge"], 4), round(frac, 4),
                    round(best["decimal_odds"], 4),
                    rl_state, str(rl_action),
                ),
            )

            # Deduct from bankroll
            current_balance = round(current_balance - bet_amount, 2)
            conn.execute(
                """
                INSERT INTO bankroll_history (event_date, action, amount, balance_after, notes)
                VALUES (?, 'bet_placed', ?, ?, ?)
                """,
                (
                    game_date, -bet_amount, current_balance,
                    f"Bet {bet_direction} on {player_name} via {best['source']}",
                ),
            )

            generated.append({
                "game_date": game_date,
                "player_name": player_name,
                "bet_direction": bet_direction,
                "bet_amount": bet_amount,
                "odds_source": best["source"],
                "implied_prob": round(best["implied_prob"], 4),
                "model_prob": round(model_prob, 4),
                "edge": round(best["source_edge"], 4),
                "kelly_fraction": round(frac, 4),
                "decimal_odds": round(best["decimal_odds"], 4),
                "status": "pending",
            })

        conn.commit()
        logger.info(f"Generated {len(generated)} paper bets for {game_date}, bankroll now ${current_balance:.2f}")
        return generated

    finally:
        conn.close()


def settle_paper_bets(game_date: str, db_path: str | Path = DB_PATH) -> dict:
    """
    Settle all pending paper bets for game_date using actual results from the results table.
    Returns summary: {settled, won, lost, net_pnl}.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        pending = conn.execute(
            """
            SELECT id, player_name, bet_direction, bet_amount, decimal_odds,
                   rl_state, rl_action
            FROM paper_bets
            WHERE game_date = ? AND status = 'pending'
            """,
            (game_date,),
        ).fetchall()

        if not pending:
            return {"settled": 0, "won": 0, "lost": 0, "net_pnl": 0.0, "game_date": game_date}

        # Load results for the date
        results = {}
        for r in conn.execute(
            "SELECT player_name, went_over FROM results WHERE game_date = ? AND went_over IS NOT NULL",
            (game_date,),
        ).fetchall():
            results[r[0].lower()] = int(r[1])

        won_count = 0
        lost_count = 0
        net_pnl = 0.0
        settled_count = 0
        current_balance = get_bankroll(db_path)
        rl_agent = RLBettingAgent.load()
        recent_acc = _get_recent_accuracy(conn)

        for bet_id, player_name, direction, amount, dec_odds, rl_state, rl_action_str in pending:
            # Try to find result (case-insensitive, partial match)
            went_over = None
            for name_key, result in results.items():
                if player_name.lower() in name_key or name_key in player_name.lower():
                    went_over = result
                    break

            if went_over is None:
                # Game not yet finished or player didn't play — leave pending
                continue

            won = (direction == "over" and went_over == 1) or (direction == "under" and went_over == 0)
            if won:
                payout = round(amount * dec_odds, 2)
                profit_loss = round(payout - amount, 2)
                won_count += 1
                current_balance = round(current_balance + payout, 2)
            else:
                payout = 0.0
                profit_loss = -amount
                lost_count += 1

            net_pnl = round(net_pnl + profit_loss, 2)

            conn.execute(
                """
                UPDATE paper_bets
                SET status = ?, payout = ?, profit_loss = ?, settled_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                ("won" if won else "lost", payout, profit_loss, bet_id),
            )
            settled_count += 1

            # RL update: reward = normalized P&L relative to bet size
            if rl_state and rl_action_str is not None:
                try:
                    rl_action = int(rl_action_str)
                    reward = profit_loss / amount if amount > 0 else 0.0
                    next_state = encode_state(
                        abs(profit_loss / amount) if amount > 0 else 0.0,
                        "medium",
                        recent_acc,
                        _time_bucket(),
                    )
                    rl_agent.update(rl_state, rl_action, reward, next_state)
                except Exception:
                    pass  # RL update is non-fatal

        if settled_count > 0:
            rl_agent.save()

            # Record net settlement in bankroll
            conn.execute(
                """
                INSERT INTO bankroll_history (event_date, action, amount, balance_after, notes)
                VALUES (?, 'settlement', ?, ?, ?)
                """,
                (
                    game_date, net_pnl, current_balance,
                    f"Settlement for {game_date}: {won_count}W/{lost_count}L",
                ),
            )
            conn.commit()

        logger.info(f"Settled {settled_count} bets for {game_date}: {won_count}W/{lost_count}L, net P&L=${net_pnl:+.2f}")
        return {
            "settled": settled_count,
            "won": won_count,
            "lost": lost_count,
            "net_pnl": net_pnl,
            "game_date": game_date,
        }

    finally:
        conn.close()


def get_trader_performance(db_path: str | Path = DB_PATH, days: int = 30) -> dict:
    """
    Aggregate performance stats from paper_bets + bankroll_history.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        import pandas as pd
        from datetime import timedelta

        cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

        bets_df = pd.read_sql_query(
            """
            SELECT game_date, odds_source, bet_direction, bet_amount,
                   decimal_odds, status, payout, profit_loss
            FROM paper_bets
            WHERE game_date >= ? AND status != 'pending'
            """,
            conn,
            params=(cutoff,),
        )

        current_balance = get_bankroll(db_path)

        # Starting balance = earliest reset or initial
        start_row = conn.execute(
            "SELECT balance_after FROM bankroll_history WHERE action IN ('reset','initial') ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        starting_balance = float(start_row[0]) if start_row else DEFAULT_BANKROLL

        # Daily balance for chart
        balance_rows = conn.execute(
            """
            SELECT event_date, balance_after
            FROM bankroll_history
            WHERE event_date >= ?
            ORDER BY created_at ASC
            """,
            (cutoff,),
        ).fetchall()
        # Keep last balance per day
        daily_map = {}
        for r in balance_rows:
            daily_map[r[0]] = float(r[1])
        daily_balance = [{"date": d, "balance": b} for d, b in sorted(daily_map.items())]

        if bets_df.empty:
            return {
                "current_balance": current_balance,
                "starting_balance": starting_balance,
                "total_pnl": round(current_balance - starting_balance, 2),
                "win_rate": None,
                "total_bets": 0,
                "total_won": 0,
                "total_lost": 0,
                "by_source": {},
                "daily_balance": daily_balance,
            }

        total = len(bets_df)
        won = int((bets_df["status"] == "won").sum())
        lost = int((bets_df["status"] == "lost").sum())
        win_rate = round(won / total, 3) if total > 0 else None

        by_source = {}
        for source, grp in bets_df.groupby("odds_source"):
            g_won = int((grp["status"] == "won").sum())
            g_total = len(grp)
            g_pnl = round(float(grp["profit_loss"].sum()), 2)
            wagered = float(grp["bet_amount"].sum())
            by_source[source] = {
                "bets": g_total,
                "won": g_won,
                "lost": g_total - g_won,
                "pnl": g_pnl,
                "roi": round(g_pnl / wagered, 3) if wagered > 0 else None,
            }

        return {
            "current_balance": round(current_balance, 2),
            "starting_balance": round(starting_balance, 2),
            "total_pnl": round(current_balance - starting_balance, 2),
            "win_rate": win_rate,
            "total_bets": total,
            "total_won": won,
            "total_lost": lost,
            "by_source": by_source,
            "daily_balance": daily_balance,
        }

    finally:
        conn.close()
