"""
Model evaluation and performance tracking.
Scores past predictions against actual results and computes calibration metrics.
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DB_PATH
from config.logging_config import setup_logging
from ingestion.nba_stats import get_yesterdays_results

logger = setup_logging("evaluate")


def score_past_predictions(db_path: str | Path = DB_PATH, game_date: str | None = None) -> dict:
    """
    Pull actual results for game_date and score any open predictions.
    Updates results table and marks predictions as correct/incorrect.
    Returns {scored: int, game_date: str}.
    """
    db_path = Path(db_path)
    game_date_str = game_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    actual_df = get_yesterdays_results(game_date_str)
    if actual_df.empty:
        logger.warning(f"No results found for {game_date_str}")
        return {"scored": 0, "game_date": game_date_str}

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    scored = 0

    try:
        for _, result_row in actual_df.iterrows():
            player_id = int(result_row["player_id"])
            actual_pts = int(result_row["actual_points"])

            # Get sportsbook line from predictions table (already stored per player)
            line_row = conn.execute(
                "SELECT sportsbook_line FROM predictions WHERE player_id = ? AND game_date = ? LIMIT 1",
                (player_id, game_date_str),
            ).fetchone()
            sb_line = line_row[0] if line_row else None
            went_over = int(actual_pts > sb_line) if sb_line is not None else None

            # Insert into results table
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO results
                    (player_id, player_name, game_date, actual_points, sportsbook_line, went_over)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        player_id,
                        result_row["player_name"],
                        game_date_str,
                        actual_pts,
                        sb_line,
                        went_over,
                    ),
                )
            except Exception as e:
                logger.debug(f"Results insert failed: {e}")

            # Mark each prediction correct/incorrect
            preds = conn.execute(
                "SELECT id, predicted_over_prob FROM predictions WHERE player_id = ? AND game_date = ?",
                (player_id, game_date_str),
            ).fetchall()

            for pred_id, pred_prob in preds:
                if went_over is None:
                    continue
                model_predicted_over = int(pred_prob > 0.5)
                model_was_correct = int(model_predicted_over == went_over)
                try:
                    conn.execute(
                        "UPDATE predictions SET model_was_correct = ? WHERE id = ?",
                        (model_was_correct, pred_id),
                    )
                    conn.execute(
                        """
                        UPDATE results SET prediction_id = ?, model_was_correct = ?
                        WHERE player_id = ? AND game_date = ?
                        """,
                        (pred_id, model_was_correct, player_id, game_date_str),
                    )
                    scored += 1
                except Exception as e:
                    logger.debug(f"Failed to update prediction correctness: {e}")

        conn.commit()
    finally:
        conn.close()

    logger.info(f"Scored {scored} predictions for {game_date_str}")
    return {"scored": scored, "game_date": game_date_str}


def analyze_prediction_errors(db_path: str | Path = DB_PATH, game_date: str | None = None) -> dict:
    """
    Deep analysis of prediction errors for a given game_date.
    Returns breakdown by confidence, model type, and worst misses.
    """
    db_path = Path(db_path)
    game_date_str = game_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        df = pd.read_sql_query(
            """
            SELECT
                p.player_name,
                p.model_type,
                p.predicted_points,
                p.confidence,
                r.actual_points,
                r.went_over,
                CASE WHEN (p.predicted_over_prob > 0.5) = r.went_over THEN 1 ELSE 0 END AS correct
            FROM predictions p
            JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
            WHERE p.game_date = ? AND r.actual_points IS NOT NULL
            """,
            conn,
            params=(game_date_str,),
        )
    finally:
        conn.close()

    if df.empty:
        return {"game_date": game_date_str, "error": "No scored predictions for this date"}

    df["error"] = df["actual_points"] - df["predicted_points"]
    df["abs_error"] = df["error"].abs()

    # Overall distribution
    error_dist = {
        "mean": round(float(df["error"].mean()), 2),
        "std": round(float(df["error"].std()), 2),
        "mae": round(float(df["abs_error"].mean()), 2),
        "count": int(len(df)),
    }

    # By confidence level
    by_confidence = {}
    for conf, grp in df.groupby("confidence"):
        by_confidence[conf] = {
            "mae": round(float(grp["abs_error"].mean()), 2),
            "accuracy": round(float(grp["correct"].mean()), 3),
            "count": int(len(grp)),
        }

    # By model type (avg across all players for that model)
    by_model = {}
    for model, grp in df.groupby("model_type"):
        by_model[model] = {
            "mae": round(float(grp["abs_error"].mean()), 2),
            "accuracy": round(float(grp["correct"].mean()), 3),
            "count": int(len(grp)),
        }

    # Worst misses (deduplicated by player — take the largest error per player)
    worst = (
        df.groupby("player_name")
        .agg(predicted_points=("predicted_points", "mean"), actual_points=("actual_points", "first"), error=("error", "mean"))
        .reset_index()
    )
    worst["abs_error"] = worst["error"].abs()
    worst = worst.nlargest(5, "abs_error")
    worst_misses = [
        {
            "player_name": row["player_name"],
            "predicted": round(float(row["predicted_points"]), 1),
            "actual": int(row["actual_points"]),
            "error": round(float(row["error"]), 1),
        }
        for _, row in worst.iterrows()
    ]

    return {
        "game_date": game_date_str,
        "error_distribution": error_dist,
        "by_confidence": by_confidence,
        "by_model": by_model,
        "worst_misses": worst_misses,
    }


def get_learning_history(db_path: str | Path = DB_PATH, days: int = 60) -> list:
    """Return retrain metrics from learning_log for the last N days."""
    db_path = Path(db_path)
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        rows = conn.execute(
            """
            SELECT retrain_date, model_name, mae, rmse, directional_accuracy, scored_predictions
            FROM learning_log
            WHERE retrain_date >= ?
            ORDER BY retrain_date ASC
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "retrain_date": r[0],
            "model_name": r[1],
            "mae": r[2],
            "rmse": r[3],
            "directional_accuracy": r[4],
            "scored_predictions": r[5],
        }
        for r in rows
    ]


def compute_model_weights(db_path: str | Path = DB_PATH, window_days: int = 14) -> dict[str, float]:
    """
    Compute normalized directional accuracy weights for each model
    over the last window_days days.

    Returns a dict like {"linear_regression": 0.28, "random_forest": 0.35, ...}
    that sums to 1.0. Falls back to equal weights if insufficient data.
    """
    db_path = Path(db_path)
    cutoff = (date.today() - timedelta(days=window_days)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        rows = conn.execute(
            """
            SELECT p.model_type,
                   AVG(CASE WHEN (p.predicted_over_prob > 0.5) = r.went_over THEN 1.0 ELSE 0.0 END) AS dir_acc,
                   COUNT(*) AS n
            FROM predictions p
            JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
            WHERE p.game_date >= ? AND r.went_over IS NOT NULL
            GROUP BY p.model_type
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        # No data yet — equal weights
        logger.debug("compute_model_weights: no data, returning equal weights")
        return {}

    # Filter models with at least 10 predictions to avoid noise
    acc_map = {r[0]: r[1] for r in rows if r[2] >= 10}
    if not acc_map:
        return {}

    # Normalize so weights sum to 1
    total = sum(acc_map.values())
    if total <= 0:
        equal = 1.0 / len(acc_map)
        return {m: equal for m in acc_map}

    return {model: round(acc / total, 6) for model, acc in acc_map.items()}


def get_model_performance(db_path: str | Path = DB_PATH, days: int = 30) -> dict:
    """
    Calculate performance metrics over recent period.
    Returns comprehensive performance breakdown.
    """
    db_path = Path(db_path)
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))

    # Overall accuracy by model type
    perf_df = pd.read_sql_query(
        """
        SELECT p.model_type, p.confidence, p.edge,
               p.predicted_over_prob, p.sportsbook_implied_prob,
               r.went_over, r.actual_points, r.sportsbook_line,
               CASE WHEN (p.predicted_over_prob > 0.5) = r.went_over THEN 1 ELSE 0 END AS correct
        FROM predictions p
        JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
        WHERE p.game_date >= ? AND r.went_over IS NOT NULL
        """,
        conn,
        params=(cutoff,),
    )
    conn.close()

    if perf_df.empty:
        return {"error": "No scored predictions yet", "days": days}

    total = len(perf_df)
    overall_acc = perf_df["correct"].mean()

    by_model = perf_df.groupby("model_type")["correct"].agg(["mean", "count"]).reset_index()
    by_model.columns = ["model_type", "accuracy", "count"]

    by_confidence = perf_df.groupby("confidence")["correct"].agg(["mean", "count"]).reset_index()
    by_confidence.columns = ["confidence", "accuracy", "count"]

    # ROI: assume $100 bet on each flagged edge (confidence != 'low')
    flagged = perf_df[perf_df["confidence"] != "low"].copy()
    bets_placed = len(flagged)
    if bets_placed > 0:
        # Typical payout at -110 is $90.91 profit on $100 bet
        winnings = flagged["correct"].sum() * 90.91
        total_wagered = bets_placed * 100
        roi = (winnings - (bets_placed - flagged["correct"].sum()) * 100) / total_wagered
    else:
        roi = 0.0

    return {
        "days": days,
        "total_predictions": total,
        "overall_accuracy": round(float(overall_acc), 4),
        "by_model": by_model.to_dict(orient="records"),
        "by_confidence": by_confidence.to_dict(orient="records"),
        "bets_placed": bets_placed,
        "roi": round(roi, 4),
    }


def get_calibration_data(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """
    Build calibration data: buckets of predicted probability vs actual hit rate.
    Perfect calibration = diagonal line.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT p.predicted_over_prob, r.went_over
        FROM predictions p
        JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
        WHERE r.went_over IS NOT NULL
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["prob_bucket", "actual_rate", "count"])

    # Create probability buckets
    df["prob_bucket"] = pd.cut(
        df["predicted_over_prob"],
        bins=[0.4, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0],
        labels=[0.45, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.95],
        include_lowest=True,
    )
    calibration = df.groupby("prob_bucket")["went_over"].agg(["mean", "count"]).reset_index()
    calibration.columns = ["prob_bucket", "actual_rate", "count"]
    calibration["prob_bucket"] = calibration["prob_bucket"].astype(float)
    return calibration


def get_market_comparison(db_path: str | Path = DB_PATH, days: int = 30) -> dict:
    """
    Compare model accuracy vs sportsbook implied accuracy vs prediction market accuracy.
    """
    db_path = Path(db_path)
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT p.predicted_over_prob, p.sportsbook_implied_prob,
               p.kalshi_prob, p.polymarket_prob,
               p.edge, p.confidence,
               r.went_over
        FROM predictions p
        JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
        WHERE p.game_date >= ? AND r.went_over IS NOT NULL
        """,
        conn,
        params=(cutoff,),
    )
    conn.close()

    if df.empty:
        return {"error": "No data yet"}

    def _accuracy(prob_col, went_over_col):
        valid = df[[prob_col, went_over_col]].dropna()
        if valid.empty:
            return None
        preds = (valid[prob_col] > 0.5).astype(int)
        return round(float((preds == valid[went_over_col]).mean()), 4)

    model_acc = _accuracy("predicted_over_prob", "went_over")
    sb_acc = _accuracy("sportsbook_implied_prob", "went_over")
    kalshi_acc = _accuracy("kalshi_prob", "went_over") if df["kalshi_prob"].notna().any() else None
    poly_acc = _accuracy("polymarket_prob", "went_over") if df["polymarket_prob"].notna().any() else None

    # Edge analysis: when model found edge, what was hit rate?
    edge_buckets = {}
    for threshold in [0.05, 0.10, 0.15]:
        edged = df[df["edge"].abs() >= threshold]
        if not edged.empty:
            model_correct = (((edged["predicted_over_prob"] > 0.5).astype(int)) == edged["went_over"]).mean()
            edge_buckets[f"edge_{int(threshold*100)}pct"] = {
                "count": len(edged),
                "model_accuracy": round(float(model_correct), 4),
            }

    return {
        "days": days,
        "model_accuracy": model_acc,
        "sportsbook_accuracy": sb_acc,
        "kalshi_accuracy": kalshi_acc,
        "polymarket_accuracy": poly_acc,
        "edge_analysis": edge_buckets,
        "total_predictions": len(df),
    }
