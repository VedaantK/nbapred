"""
Model training entrypoint.
Run after backfill_data.py has populated the database.

Usage:
    cd nba-predictor
    python train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.db import init_db
from config.settings import DB_PATH
from config.logging_config import setup_logging
from models.train import train_all_models

logger = setup_logging("train_entrypoint")


def main():
    init_db(DB_PATH)
    logger.info("Starting model training...")
    results = train_all_models(DB_PATH)
    if not results:
        logger.error("Training failed or no data available.")
        return

    logger.info("\n=== Training Results ===")
    for name, info in results.items():
        m = info["metrics"]
        logger.info(f"{name}: MAE={m['mae']} pts | RMSE={m['rmse']} pts | Dir.Acc={m['directional_accuracy']:.1%}")

    logger.info("\nModels saved. Ready to run daily_pipeline.py")


if __name__ == "__main__":
    main()
