"""
One-time historical data backfill.
Run this FIRST before training any models.

Pulls game logs for top ~150 players for current + last season,
plus team defensive stats. Takes 15-30 min due to NBA API rate limits.

Usage:
    cd nba-predictor
    python -m scripts.backfill_data
"""

import sqlite3
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.db import init_db
from config.settings import (DB_PATH, NBA_SEASONS, TOP_PLAYERS_COUNT, REQUEST_DELAY,
                             MIN_MINUTES_FOR_RANKING)
from config.logging_config import setup_logging
from ingestion.nba_stats import (
    get_player_game_logs,
    get_team_defensive_stats,
    get_player_usage_rates,
    save_game_logs_to_db,
    save_team_stats_to_db,
)

logger = setup_logging("backfill")


def get_top_players(n: int = TOP_PLAYERS_COUNT) -> list[dict]:
    """Get top N players by usage — approximately the most relevant players.

    The player list comes from the usage endpoint itself (live), not from
    nba_api's bundled static player list, which lags a season behind and
    is missing current rookies.
    """
    # Rank by usage among real rotation players. Two earlier failure modes:
    #  1. Any error here used to fall back to players[:n]. get_active_players()
    #     is alphabetical, so the whole DB silently became players named A–D.
    #     There is no safe fallback, so fail loudly instead.
    #  2. Raw usage rate is topped by two-way players with a few garbage-time
    #     minutes (50% usage on 6 MPG), who have no prop markets. Filter first.
    usage_df = get_player_usage_rates(NBA_SEASONS[0])
    if usage_df.empty or "usage_rate" not in usage_df.columns:
        raise RuntimeError(
            f"Could not fetch usage rates for {NBA_SEASONS[0]} from stats.nba.com; "
            "refusing to fall back to an alphabetical player list. Re-run later."
        )

    if {"games_played", "minutes"} <= set(usage_df.columns):
        # Early in a season nobody has many games, so scale the games bar to the
        # league leader (e.g. 10 games played -> need 4).
        min_games = max(3, int(0.4 * usage_df["games_played"].max()))
        rotation = usage_df[(usage_df["games_played"] >= min_games)
                            & (usage_df["minutes"] >= MIN_MINUTES_FOR_RANKING)]
        logger.info(f"{len(rotation)} rotation players (>= {min_games} GP, "
                    f">= {MIN_MINUTES_FOR_RANKING} MPG) of {len(usage_df)}")
        usage_df = rotation

    usage_df = usage_df.sort_values("usage_rate", ascending=False)
    top = usage_df.head(n)
    return [{"id": int(pid), "full_name": name}
            for pid, name in zip(top["player_id"], top["player_name"])]


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


def _seasons_loaded(player_id: int) -> set[str]:
    """Seasons that already have game log rows for this player."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT season FROM player_game_logs WHERE player_id = ?", (player_id,)
        ).fetchall()
    return {r[0] for r in rows}


def backfill_player_logs(players: list[dict]):
    """Pull game logs for each player across all seasons."""
    total = len(players)
    for idx, player in enumerate(players):
        player_id = player["id"]
        player_name = player["full_name"]
        logger.info(f"[{idx+1}/{total}] {player_name}")

        # Finished seasons never change, so skip any already in the DB. The
        # current season (NBA_SEASONS[0]) is always re-pulled for new games.
        done = _seasons_loaded(player_id)
        for season in NBA_SEASONS:
            if season != NBA_SEASONS[0] and season in done:
                continue
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
