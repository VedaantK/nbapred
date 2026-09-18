"""
NBA game logs and player stats via nba_api.
Pulls player game logs, team defensive stats, and schedule data.
"""

import time
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DB_PATH, REQUEST_DELAY, NBA_SEASONS
from config.logging_config import setup_logging

logger = setup_logging("nba_stats")


def _sleep():
    time.sleep(REQUEST_DELAY)


def get_active_players() -> list[dict]:
    """Return list of active NBA players with their IDs."""
    from nba_api.stats.static import players
    return players.get_active_players()


def get_player_game_logs(player_id: int, season: str = "2024-25") -> pd.DataFrame:
    """
    Pull full game log for a player in a given season.
    Returns DataFrame with normalized column names.
    """
    from nba_api.stats.endpoints import PlayerGameLog

    try:
        _sleep()
        log = PlayerGameLog(player_id=player_id, season=season, timeout=20)
        df = log.get_data_frames()[0]
    except Exception as e:
        logger.warning(f"Failed to fetch game log for player {player_id} season {season}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    rename = {
        "GAME_DATE": "game_date",
        "Game_ID": "game_id",
        "MATCHUP": "matchup",
        "WL": "wl",
        "MIN": "minutes",
        "PTS": "points",
        "FGM": "fgm",
        "FGA": "fga",
        "FG_PCT": "fg_pct",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FT_PCT": "ft_pct",
        "FTM": "ftm",
        "FTA": "fta",
        "REB": "rebounds",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "TOV": "turnovers",
        "PLUS_MINUS": "plus_minus",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Normalize game_date to YYYY-MM-DD.
    #
    # The API returns "Apr 12, 2026" — abbreviated month. Left to infer, pandas
    # reads the first row to pick a format, and a season whose first row falls
    # in May ("May 16, 2021") looks like a full month name (%B), so every
    # abbreviated row after it raises. That killed the whole backfill on the
    # 2020-21 season. Parse the known format explicitly instead.
    parsed = pd.to_datetime(df["game_date"], format="%b %d, %Y", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(df["game_date"], format="mixed", errors="coerce")
    df["game_date"] = parsed.dt.strftime("%Y-%m-%d")

    # Derive is_home from matchup string
    df["is_home"] = df["matchup"].apply(lambda m: 1 if "vs." in str(m) else 0)

    df["season"] = season
    df["player_id"] = player_id

    return df


def _season_end_date(season: str) -> str:
    """'2024-25' -> '2025-06-30'. Dates a season's snapshot to when it finished."""
    try:
        return f"{int(season.split('-')[0]) + 1}-06-30"
    except (ValueError, IndexError):
        return date.today().strftime("%Y-%m-%d")


def get_team_defensive_stats(season: str = "2024-25") -> pd.DataFrame:
    """
    Pull team-level defensive rating, pace, and opponent points allowed.

    Two calls are needed: DEF_RATING and PACE live under the Advanced measure
    type, but points allowed does not — it is OPP_PTS under Opponent. (The old
    OPP_PTS_OFF_TOV mapping was both absent from Advanced and the wrong stat:
    points off turnovers, not points allowed.)
    """
    from nba_api.stats.endpoints import LeagueDashTeamStats
    from nba_api.stats.static import teams as _static_teams

    try:
        _sleep()
        adv = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
    except Exception as e:
        logger.warning(f"Failed to fetch team advanced stats for {season}: {e}")
        return pd.DataFrame()

    if adv.empty:
        return adv

    df = adv.rename(columns={
        "TEAM_ID": "team_id",
        "DEF_RATING": "defensive_rating",
        "PACE": "pace",
    })

    # Opponent points allowed — separate measure type, non-fatal if it fails.
    try:
        _sleep()
        opp = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        if "OPP_PTS" in opp.columns:
            opp = opp.rename(columns={"TEAM_ID": "team_id", "OPP_PTS": "opp_points_allowed"})
            df = df.merge(opp[["team_id", "opp_points_allowed"]], on="team_id", how="left")
    except Exception as e:
        logger.warning(f"Failed to fetch opponent points for {season}: {e}")

    if "opp_points_allowed" not in df.columns:
        df["opp_points_allowed"] = np.nan

    # LeagueDashTeamStats returns TEAM_NAME, never TEAM_ABBREVIATION. Without
    # this map every row stored a blank abbreviation and the join in
    # features/engineer.py — which matches on "GSW"-style codes parsed out of
    # the matchup string — silently matched nothing.
    abbr_by_id = {t["id"]: t["abbreviation"] for t in _static_teams.get_teams()}
    df["team_abbreviation"] = df["team_id"].map(abbr_by_id)

    df["season"] = season
    # Date each snapshot to the end of ITS season rather than today. Two
    # reasons: team_stats is UNIQUE(team_id, game_date), so writing every
    # season with today's date meant only the first season backfilled survived;
    # and a season-stamped row lets the feature builder pick a snapshot that
    # predates the game it is describing.
    df["game_date"] = _season_end_date(season)

    keep = ["team_id", "team_abbreviation", "season", "game_date",
            "defensive_rating", "pace", "opp_points_allowed"]
    return df[[c for c in keep if c in df.columns]]


def get_todays_games() -> list[dict]:
    """Get today's NBA schedule."""
    from nba_api.stats.endpoints import ScoreboardV2

    try:
        _sleep()
        scoreboard = ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"), timeout=15)
        games_df = scoreboard.get_data_frames()[0]
    except Exception as e:
        logger.warning(f"Failed to fetch today's games: {e}")
        return []

    if games_df.empty:
        return []

    games = []
    for _, row in games_df.iterrows():
        games.append({
            "game_id": row.get("GAME_ID", ""),
            "home_team": row.get("HOME_TEAM_ABBREVIATION", row.get("HOME_TEAM_ID", "")),
            "away_team": row.get("VISITOR_TEAM_ABBREVIATION", row.get("VISITOR_TEAM_ID", "")),
            "game_date": date.today().strftime("%Y-%m-%d"),
        })
    return games


def get_player_usage_rates(season: str = "2024-25") -> pd.DataFrame:
    """Pull usage rate data for all players."""
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    try:
        _sleep()
        stats = LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Usage",
            per_mode_detailed="PerGame",
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        logger.warning(f"Failed to fetch usage rates for {season}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    rename = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "USG_PCT": "usage_rate",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df[["player_id", "player_name", "usage_rate"]] if "usage_rate" in df.columns else df


def get_yesterdays_results(game_date: str | None = None) -> pd.DataFrame:
    """
    Pull actual player point totals for a given date (defaults to yesterday).
    Uses LeagueGameFinder + BoxScoreTraditionalV3 — updates immediately after
    games finish (LeagueDashPlayerStats lags 1-2 days).
    Returns DataFrame with columns: player_id, player_name, actual_points, game_date.
    """
    from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreTraditionalV3
    from datetime import timedelta

    target = game_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Step 1: find all game IDs played on target date
    try:
        _sleep()
        gf = LeagueGameFinder(
            date_from_nullable=target,
            date_to_nullable=target,
            league_id_nullable="00",
            timeout=20,
        )
        games_df = gf.get_data_frames()[0]
    except Exception as e:
        logger.warning(f"Failed to fetch game IDs for {target}: {e}")
        return pd.DataFrame()

    if games_df.empty:
        logger.warning(f"No games found for {target}")
        return pd.DataFrame()

    game_ids = games_df["GAME_ID"].drop_duplicates().tolist()
    logger.info(f"Found {len(game_ids)} game IDs for {target}, fetching box scores...")

    # Step 2: fetch box scores and collect player points
    rows = []
    seen_players: set[int] = set()

    for game_id in game_ids:
        try:
            _sleep()
            bs = BoxScoreTraditionalV3(game_id=game_id, timeout=20)
            df = bs.get_data_frames()[0]
        except Exception as e:
            logger.debug(f"BoxScore fetch failed for game {game_id}: {e}")
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            pid = row.get("personId")
            pts = row.get("points")
            if pid is None or pts is None:
                continue
            pid = int(pid)
            if pid in seen_players:
                continue  # dedup (both teams in same game_id response)

            # Skip DNPs — minutes string is empty or zero when player didn't play
            minutes_raw = str(row.get("minutes") or "")
            played = bool(minutes_raw and minutes_raw not in ("PT00M00.00S", "PT0M0.00S", ""))
            if not played:
                continue

            seen_players.add(pid)

            first = str(row.get("firstName", "")).strip()
            last  = str(row.get("familyName", "")).strip()
            full_name = f"{first} {last}".strip()

            rows.append({
                "player_id": pid,
                "player_name": full_name,
                "actual_points": int(pts),
                "game_date": target,
            })

    if not rows:
        logger.warning(f"No player stats extracted for {target}")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    logger.info(f"Fetched actual points for {len(result)} players on {target}")
    return result


def save_game_logs_to_db(df: pd.DataFrame, player_id: int, player_name: str, db_path: str | Path = DB_PATH):
    """Insert/update player game logs into SQLite. Uses INSERT OR IGNORE to avoid duplicates."""
    if df.empty:
        return

    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    inserted = 0

    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO player_game_logs
                (player_id, player_name, team_abbreviation, game_id, game_date, season,
                 matchup, is_home, wl, minutes, points, fgm, fga, fg_pct, fg3m, fg3a,
                 ftm, fta, rebounds, assists, steals, blocks, turnovers, plus_minus)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    player_id,
                    player_name,
                    row.get("team_abbreviation", row.get("TEAM_ABBREVIATION", "")),
                    row.get("game_id", ""),
                    row.get("game_date", ""),
                    row.get("season", ""),
                    row.get("matchup", ""),
                    row.get("is_home", 0),
                    row.get("wl", ""),
                    row.get("minutes", 0),
                    row.get("points", 0),
                    row.get("fgm", 0),
                    row.get("fga", 0),
                    row.get("fg_pct", 0),
                    row.get("fg3m", 0),
                    row.get("fg3a", 0),
                    row.get("ftm", 0),
                    row.get("fta", 0),
                    row.get("rebounds", 0),
                    row.get("assists", 0),
                    row.get("steals", 0),
                    row.get("blocks", 0),
                    row.get("turnovers", 0),
                    row.get("plus_minus", 0),
                ),
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"Skipping row: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved {inserted} game log rows for {player_name}")


def save_team_stats_to_db(df: pd.DataFrame, db_path: str | Path = DB_PATH):
    """Insert/update team stats into SQLite."""
    if df.empty:
        return

    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))

    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO team_stats
                (team_id, team_abbreviation, season, game_date, defensive_rating, pace, opp_points_allowed)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    row.get("team_id", 0),
                    row.get("team_abbreviation", ""),
                    row.get("season", ""),
                    row.get("game_date", ""),
                    row.get("defensive_rating"),
                    row.get("pace"),
                    row.get("opp_points_allowed"),
                ),
            )
        except Exception as e:
            logger.debug(f"Skipping team stats row: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved {len(df)} team stat rows")
