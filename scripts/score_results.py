"""
Score predictions against actual results, analyze errors, and retrain models.
Run this every morning after games finish (e.g., 8:00 AM).

Usage:
    cd nba-predictor
    python -m scripts.score_results
    python -m scripts.score_results --date 2026-03-27
"""

import sys
import sqlite3
import argparse
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH
from config.logging_config import setup_logging
from models.evaluate import score_past_predictions, get_model_performance

logger = setup_logging("score_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date to score (YYYY-MM-DD, default: yesterday)")
    args = parser.parse_args()

    game_date = args.date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"=== Scoring predictions for {game_date} ===")

    # Score predictions
    result = score_past_predictions(DB_PATH, game_date)
    scored = result.get("scored", 0)
    logger.info(f"Scored {scored} predictions")

    # Performance summary
    perf = get_model_performance(DB_PATH, days=7)
    if "error" not in perf:
        logger.info("\n=== 7-Day Performance ===")
        logger.info(f"Overall accuracy: {perf.get('overall_accuracy', 0):.1%}")
        logger.info(f"Total predictions: {perf.get('total_predictions', 0)}")
        logger.info(f"ROI: {perf.get('roi', 0):.1%}")
        for m in perf.get("by_model", []):
            logger.info(f"  {m['model_type']}: {m['accuracy']:.1%} ({m['count']} predictions)")
        for c in perf.get("by_confidence", []):
            logger.info(f"  Confidence={c['confidence']}: {c['accuracy']:.1%} ({c['count']} predictions)")
    else:
        logger.info(f"Performance: {perf['error']}")

    # Always retrain after scoring
    logger.info("\n=== Retraining Models ===")
    from models.train import train_all_models
    train_results = train_all_models(DB_PATH)
    if train_results:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            for name, info in train_results.items():
                m = info["metrics"]
                logger.info(f"{name}: MAE={m['mae']}, RMSE={m['rmse']}, DirAcc={m['directional_accuracy']:.1%}")
                conn.execute(
                    """
                    INSERT INTO learning_log
                    (retrain_date, scored_predictions, model_name, mae, rmse, directional_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (game_date, scored, name, m["mae"], m["rmse"], m["directional_accuracy"]),
                )
            conn.commit()
        finally:
            conn.close()
        logger.info("Retraining complete — metrics saved to learning_log")
    else:
        logger.warning("Retraining produced no results (insufficient data?)")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
