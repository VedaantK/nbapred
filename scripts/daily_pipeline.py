"""
Daily pipeline — run a few hours before games start.

Steps:
1. Update player game logs (new games since last run)
2. Update team stats
3. Fetch sportsbook lines for today's games
4. Fetch Kalshi markets
5. Fetch Polymarket markets
6. Generate predictions
7. Log summary

Usage:
    cd nba-predictor
    python -m scripts.daily_pipeline

Schedule (Windows Task Scheduler):
    Action: python -m scripts.daily_pipeline
    Start in: C:\\path\\to\\nba-predictor
    Trigger: Daily at 11:00 AM Eastern
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.db import init_db
from config.settings import DB_PATH, NBA_SEASONS, REQUEST_DELAY
from config.logging_config import setup_logging
from ingestion.nba_stats import (
    get_active_players,
    get_player_game_logs,
    get_team_defensive_stats,
    save_game_logs_to_db,
    save_team_stats_to_db,
)
from ingestion.odds_api import fetch_and_store_all_props
from ingestion.kalshi import fetch_and_store_kalshi_markets
from ingestion.polymarket import fetch_and_store_polymarket_markets
from models.predict import predict_today

logger = setup_logging("daily_pipeline")


def update_recent_game_logs():
    """Pull game logs for all tracked players to pick up new games."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    tracked = conn.execute(
        "SELECT DISTINCT player_id, player_name FROM player_game_logs"
    ).fetchall()
    conn.close()

    logger.info(f"Updating game logs for {len(tracked)} tracked players...")
    for player_id, player_name in tracked:
        df = get_player_game_logs(player_id, NBA_SEASONS[0])
        if not df.empty:
            save_game_logs_to_db(df, player_id, player_name, DB_PATH)


def main():
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"=== Daily Pipeline: {today} ===")

    # Ensure DB is initialized
    init_db(DB_PATH)

    # 1. Update game logs
    logger.info("Step 1: Updating player game logs...")
    update_recent_game_logs()

    # 2. Update team stats
    logger.info("Step 2: Updating team defensive stats...")
    df = get_team_defensive_stats(NBA_SEASONS[0])
    if not df.empty:
        save_team_stats_to_db(df, DB_PATH)

    # 3. Sportsbook lines
    logger.info("Step 3: Fetching sportsbook lines...")
    fetch_and_store_all_props(DB_PATH)

    # 4. Kalshi
    logger.info("Step 4: Fetching Kalshi markets...")
    fetch_and_store_kalshi_markets(DB_PATH)

    # 5. Polymarket
    logger.info("Step 5: Fetching Polymarket markets...")
    fetch_and_store_polymarket_markets(DB_PATH)

    # 6. Generate predictions
    logger.info("Step 6: Generating predictions...")
    predictions = predict_today(DB_PATH)

    # 7. Summary
    if predictions:
        logger.info(f"\n=== PREDICTIONS SUMMARY ({today}) ===")
        logger.info(f"Total predictions: {len(predictions)}")
        high_conf = [p for p in predictions if p["confidence"] == "high"]
        med_conf = [p for p in predictions if p["confidence"] == "medium"]
        logger.info(f"High confidence: {len(high_conf)}")
        logger.info(f"Medium confidence: {len(med_conf)}")
        if predictions:
            best = predictions[0]
            logger.info(f"Best edge: {best['player_name']} — {best['edge']*100:.1f}% edge @ {best['sportsbook_line']} pts")
    else:
        logger.info("No predictions generated (no games today or missing data)")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
