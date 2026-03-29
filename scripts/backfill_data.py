"""
One-time historical data backfill.
Run this FIRST before training any models.

Pulls game logs for top ~150 players for current + last season,
plus team defensive stats. Takes 15-30 min due to NBA API rate limits.

Usage:
    cd nba-predictor
    python -m scripts.backfill_data
"""

import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.db import init_db
from config.settings import DB_PATH, NBA_SEASONS, TOP_PLAYERS_COUNT, REQUEST_DELAY
from config.logging_config import setup_logging
from ingestion.nba_stats import (
    get_active_players,
    get_player_game_logs,
    get_team_defensive_stats,
    get_player_usage_rates,
    save_game_logs_to_db,
    save_team_stats_to_db,
)

logger = setup_logging("backfill")


def get_top_players(n: int = TOP_PLAYERS_COUNT) -> list[dict]:
    """Get top N players by usage — approximately the most relevant players."""
    logger.info("Fetching active player list...")
    players = get_active_players()
    logger.info(f"Found {len(players)} active players")

    # Pull usage stats to rank players
    try:
        import pandas as pd
        usage_df = get_player_usage_rates(NBA_SEASONS[0])
        if not usage_df.empty and "player_id" in usage_df.columns:
            top_ids = set(usage_df.head(n)["player_id"].tolist())
            top_players = [p for p in players if p["id"] in top_ids]
            if top_players:
                return top_players
    except Exception as e:
        logger.warning(f"Couldn't rank by usage: {e} — using first {n} players")

    return players[:n]


def backfill_team_stats():
    """Pull team defensive stats for all configured seasons."""
    for season in NBA_SEASONS:
        logger.info(f"Fetching team stats for {season}...")
        df = get_team_defensive_stats(season)
        if not df.empty:
            save_team_stats_to_db(df, DB_PATH)
        else:
            logger.warning(f"No team stats returned for {season}")
        time.sleep(REQUEST_DELAY)


def backfill_player_logs(players: list[dict]):
    """Pull game logs for each player across all seasons."""
    total = len(players)
    for idx, player in enumerate(players):
        player_id = player["id"]
        player_name = player["full_name"]
        logger.info(f"[{idx+1}/{total}] {player_name}")

        for season in NBA_SEASONS:
            df = get_player_game_logs(player_id, season)
            if not df.empty:
                save_game_logs_to_db(df, player_id, player_name, DB_PATH)
            else:
                logger.debug(f"  No data for {player_name} in {season}")


def main():
    logger.info("=== NBA Predictor Backfill Started ===")
    logger.info(f"Database: {DB_PATH}")

    # Initialize DB
    init_db(DB_PATH)

    # Team stats first (fast)
    backfill_team_stats()

    # Player game logs (slow — ~150 players × 2 seasons)
    players = get_top_players(TOP_PLAYERS_COUNT)
    logger.info(f"Backfilling game logs for {len(players)} players across {len(NBA_SEASONS)} seasons...")
    logger.info("Estimated time: 15-30 minutes (NBA API rate limits)")

    backfill_player_logs(players)

    logger.info("=== Backfill Complete ===")
    logger.info("Next step: python -m models.train")


if __name__ == "__main__":
    main()
