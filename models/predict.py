"""
Generate predictions for upcoming games.
Uses trained models + historical residuals to estimate over/under probabilities.
"""

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DB_PATH, MIN_GAMES_FOR_PREDICTION, EDGE_THRESHOLD
from config.logging_config import setup_logging
from features.engineer import build_features_for_player
from ingestion.nba_stats import get_active_players
from models.train import load_models, load_residuals, load_feature_names, load_model_weights
from utils.name_matcher import find_best_match

logger = setup_logging("predict")


def calculate_over_probability(
    predicted_points: float,
    line: float,
    model_residuals: np.ndarray,
) -> float:
    """
    Estimate P(actual > line) using the model's prediction + empirical residual distribution.
    """
    if model_residuals is None or len(model_residuals) == 0:
        diff = predicted_points - line
        return float(1 / (1 + np.exp(-diff * 0.3)))

    simulated = predicted_points + model_residuals
    return float(np.mean(simulated > line))


def _get_sportsbook_line(player_name: str, game_date: str, conn: sqlite3.Connection) -> tuple[float | None, float | None]:
    """Fetch best available sportsbook line for a player on game_date."""
    all_names = conn.execute(
        "SELECT DISTINCT player_name FROM sportsbook_lines WHERE game_date = ?", (game_date,)
    ).fetchall()
    candidate_names = [r[0] for r in all_names]
    best_name = find_best_match(player_name, candidate_names)

    if best_name is None:
        return None, None

    row = conn.execute(
        """
        SELECT line, implied_over_prob FROM sportsbook_lines
        WHERE player_name = ? AND game_date = ?
        ORDER BY CASE bookmaker WHEN 'draftkings' THEN 1 WHEN 'fanduel' THEN 2 ELSE 3 END
        LIMIT 1
        """,
        (best_name, game_date),
    ).fetchone()

    if row:
        return row[0], row[1]
    return None, None


def _get_market_probs(player_name: str, game_date: str, conn: sqlite3.Connection) -> tuple[float | None, float | None]:
    """Fetch Kalshi and Polymarket probabilities for a player."""
    all_names = conn.execute(
        "SELECT DISTINCT player_name FROM prediction_market_lines WHERE game_date = ?", (game_date,)
    ).fetchall()
    candidate_names = [r[0] for r in all_names]
    best_name = find_best_match(player_name, candidate_names)

    if best_name is None:
        return None, None

    rows = conn.execute(
        """
        SELECT source, over_prob FROM prediction_market_lines
        WHERE player_name = ? AND game_date = ?
        """,
        (best_name, game_date),
    ).fetchall()

    kalshi_prob = None
    poly_prob = None
    for row in rows:
        if row[0] == "kalshi":
            kalshi_prob = row[1]
        elif row[0] == "polymarket":
            poly_prob = row[1]

    return kalshi_prob, poly_prob


def _assign_confidence(model_probs: list[float], edge: float) -> str:
    """Assign confidence level based on model agreement and edge size."""
    if not model_probs:
        return "low"

    over_votes = sum(1 for p in model_probs if p > 0.5)
    under_votes = len(model_probs) - over_votes
    majority = max(over_votes, under_votes)

    if majority == len(model_probs) and abs(edge) >= 0.12:
        return "high"
    if majority >= 3 and abs(edge) >= EDGE_THRESHOLD:
        return "medium"
    return "low"


def predict_today(db_path: str | Path = DB_PATH, game_date: str | None = None) -> list[dict]:
    """
    Generate predictions for all players with sportsbook lines on game_date (default today).
    Returns list of prediction dicts sorted by edge (largest first).
    """
    db_path = Path(db_path)
    today = game_date or date.today().strftime("%Y-%m-%d")

    models = load_models()
    if not models:
        logger.error("No trained models found. Run train.py first.")
        return []

    residuals = load_residuals()
    feature_names = load_feature_names()
    model_weights = load_model_weights()  # {} if not computed yet → falls back to equal

    # get_active_players() reads from a bundled static JSON — no HTTP call
    active_players = get_active_players()
    player_name_to_id = {p["full_name"]: p["id"] for p in active_players}
    all_nba_names = list(player_name_to_id.keys())

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        # Prefer Kalshi-listed players; fall back to sportsbook players if Kalshi is empty.
        kalshi_rows = conn.execute(
            "SELECT DISTINCT player_name FROM prediction_market_lines "
            "WHERE game_date = ? AND source = 'kalshi'",
            (today,),
        ).fetchall()
        lined_player_names = [r[0] for r in kalshi_rows]

        if lined_player_names:
            logger.info(f"Predicting {len(lined_player_names)} Kalshi-listed players for {today}")
        else:
            sb_rows = conn.execute(
                "SELECT DISTINCT player_name FROM sportsbook_lines WHERE game_date = ?",
                (today,),
            ).fetchall()
            lined_player_names = [r[0] for r in sb_rows]
            if lined_player_names:
                logger.info(
                    f"No Kalshi data for {today} — falling back to "
                    f"{len(lined_player_names)} sportsbook players"
                )

        if not lined_player_names:
            logger.warning(f"No lined players for {today} — run pipeline to fetch odds first")
            return []

        predictions = []

        for lined_name in lined_player_names:
            # Match lined player name to NBA API player
            nba_name = find_best_match(lined_name, all_nba_names)
            if nba_name is None:
                logger.debug(f"No NBA match for '{lined_name}'")
                continue

            player_id = player_name_to_id[nba_name]

            # Check we have enough game history
            count = conn.execute(
                "SELECT COUNT(*) FROM player_game_logs WHERE player_id = ?", (player_id,)
            ).fetchone()[0]
            if count < MIN_GAMES_FOR_PREDICTION:
                logger.debug(f"Skipping {nba_name}: only {count} games in DB")
                continue

            # Build features (opens+closes its own read-only connection)
            try:
                features = build_features_for_player(
                    player_id=player_id,
                    game_date=today,
                    db_path=db_path,
                )
            except Exception as e:
                logger.warning(f"Feature build failed for {nba_name}: {e}")
                continue

            if not features:
                continue

            # Align features to training order
            feature_vector = pd.DataFrame([{f: features.get(f, np.nan) for f in feature_names}])

            # Get sportsbook line (may be None if player is PM-only)
            sb_line, sb_implied_prob = _get_sportsbook_line(lined_name, today, conn)

            # Get prediction market probs
            kalshi_prob, poly_prob = _get_market_probs(lined_name, today, conn)

            # If no sportsbook line, use prediction market line as the over/under threshold
            if sb_line is None:
                pm_row = conn.execute(
                    """
                    SELECT line, over_prob FROM prediction_market_lines
                    WHERE game_date = ? AND player_name = ?
                    ORDER BY source ASC LIMIT 1
                    """,
                    (today, lined_name),
                ).fetchone()
                if pm_row and pm_row[0] is not None:
                    sb_line = pm_row[0]
                    sb_implied_prob = pm_row[1]
                else:
                    continue  # no line at all — can't make a prediction

            model_results = {}

            for model_name, pipeline in models.items():
                try:
                    fv = feature_vector.values if hasattr(feature_vector, "values") else feature_vector
                    pred_pts = float(pipeline.predict(fv)[0])
                    model_residuals = residuals.get(model_name)
                    over_prob = calculate_over_probability(pred_pts, sb_line, model_residuals)
                    model_results[model_name] = {"predicted_points": pred_pts, "over_prob": over_prob}
                except Exception as e:
                    logger.warning(f"Prediction failed for {nba_name} ({model_name}): {e}")

            if not model_results:
                continue

            # Dynamic weighted ensemble — fall back to equal weights when no weights saved
            active_weights = {m: model_weights.get(m) for m in model_results if model_weights.get(m) is not None}
            if active_weights:
                total_w = sum(active_weights.values())
                w_pts = sum(model_results[m]["predicted_points"] * w for m, w in active_weights.items()) / total_w
                w_prob = sum(model_results[m]["over_prob"] * w for m, w in active_weights.items()) / total_w
                avg_pred_pts = w_pts
                avg_over_prob = w_prob
            else:
                avg_pred_pts = np.mean([r["predicted_points"] for r in model_results.values()])
                avg_over_prob = np.mean([r["over_prob"] for r in model_results.values()])

            model_probs = [r["over_prob"] for r in model_results.values()]
            edge = avg_over_prob - (sb_implied_prob or 0.5)
            confidence = _assign_confidence(model_probs, edge)

            pred = {
                "player_id": player_id,
                "player_name": nba_name,
                "player_name_display": lined_name,
                "game_date": today,
                "predicted_points": int(round(avg_pred_pts)),
                "predicted_over_prob": round(avg_over_prob, 4),
                "sportsbook_line": sb_line,
                "sportsbook_implied_prob": sb_implied_prob,
                "kalshi_prob": kalshi_prob,
                "polymarket_prob": poly_prob,
                "edge": round(edge, 4),
                "confidence": confidence,
                "model_breakdown": {
                    k: {"predicted_points": round(v["predicted_points"], 1), "over_prob": round(v["over_prob"], 4)}
                    for k, v in model_results.items()
                },
            }
            predictions.append(pred)

            # Store in DB
            for model_name, model_res in model_results.items():
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO predictions
                        (player_id, player_name, game_date, predicted_points, predicted_over_prob,
                         sportsbook_line, sportsbook_implied_prob, kalshi_prob, polymarket_prob,
                         model_type, edge, confidence)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            player_id, nba_name, today,
                            model_res["predicted_points"], model_res["over_prob"],
                            sb_line, sb_implied_prob, kalshi_prob, poly_prob,
                            model_name, edge, confidence,
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to store prediction: {e}")

        conn.commit()
    finally:
        conn.close()

    predictions.sort(key=lambda x: abs(x["edge"]), reverse=True)
    logger.info(f"Generated {len(predictions)} predictions for {today}")
    return predictions
