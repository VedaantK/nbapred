"""
Injury and availability tracking via ESPN public API.
No API key required. Falls back gracefully on network errors.

ESPN endpoint: site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries
"""

import sqlite3
from datetime import date
from pathlib import Path

import requests

from config.settings import DB_PATH
from config.logging_config import setup_logging

logger = setup_logging("injuries")

# ESPN team IDs mapped to abbreviations (all 30 NBA teams)
ESPN_TEAMS = {
    1: "ATL", 2: "BOS", 3: "NOP", 4: "CHI", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GSW", 10: "HOU", 11: "IND", 12: "LAC",
    13: "LAL", 14: "MIA", 15: "MIL", 16: "MIN", 17: "BKN", 18: "NYK",
    19: "ORL", 20: "PHI", 21: "PHX", 22: "POR", 23: "SAC", 24: "SAS",
    25: "OKC", 26: "UTA", 27: "MEM", 28: "WAS", 29: "TOR", 30: "CHA",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries"
REQUEST_TIMEOUT = 10  # seconds per team request


def _fetch_team_injuries(team_id: int) -> list[dict]:
    """Fetch injury report for one team from ESPN. Returns [] on error."""
    url = ESPN_BASE.format(team_id=team_id)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"ESPN fetch failed for team {team_id}: {e}")
        return []

    players = []
    for item in data.get("injuries", []):
        athlete = item.get("athlete", {})
        player_id = athlete.get("id")
        if not player_id:
            continue
        players.append({
            "player_id": int(player_id),
            "player_name": athlete.get("displayName", ""),
            "team_id": team_id,
            "team_abbreviation": ESPN_TEAMS.get(team_id, ""),
            "status": item.get("status", ""),
            "description": item.get("longComment") or item.get("shortComment") or "",
        })
    return players


def fetch_and_store_injuries(db_path: str | Path = DB_PATH, fetch_date: str | None = None) -> int:
    """
    Fetch injury reports for all 30 teams and store them in injury_status.
    Returns the count of injured/questionable players stored.
    """
    db_path = Path(db_path)
    today = fetch_date or date.today().strftime("%Y-%m-%d")

    all_injured = []
    for team_id in ESPN_TEAMS:
        players = _fetch_team_injuries(team_id)
        all_injured.extend(players)

    if not all_injured:
        logger.warning("No injury data retrieved from ESPN")
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    stored = 0
    try:
        for p in all_injured:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO injury_status
                    (player_id, player_name, team_id, team_abbreviation, status, description, fetched_date)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        p["player_id"], p["player_name"], p["team_id"],
                        p["team_abbreviation"], p["status"], p["description"], today,
                    ),
                )
                stored += 1
            except Exception as e:
                logger.debug(f"Failed to store injury for {p.get('player_name')}: {e}")
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Stored {stored} injury records for {today}")
    return stored


def get_injured_players(db_path: str | Path = DB_PATH, fetch_date: str | None = None) -> dict[int, dict]:
    """
    Return a dict of {player_id: {status, description, team_abbreviation}}
    for players who are OUT or QUESTIONABLE on fetch_date.
    """
    db_path = Path(db_path)
    today = fetch_date or date.today().strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        rows = conn.execute(
            """
            SELECT player_id, player_name, team_abbreviation, status, description
            FROM injury_status
            WHERE fetched_date = ?
              AND UPPER(status) IN ('OUT', 'DOUBTFUL', 'QUESTIONABLE', 'DAY-TO-DAY')
            """,
            (today,),
        ).fetchall()
    finally:
        conn.close()

    return {
        row[0]: {
            "player_name": row[1],
            "team_abbreviation": row[2],
            "status": row[3],
            "description": row[4],
        }
        for row in rows
    }


def get_team_injury_report(team_abbreviation: str, db_path: str | Path = DB_PATH, fetch_date: str | None = None) -> list[dict]:
    """Return all injury records for a specific team on fetch_date."""
    db_path = Path(db_path)
    today = fetch_date or date.today().strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        rows = conn.execute(
            """
            SELECT player_id, player_name, status, description
            FROM injury_status
            WHERE team_abbreviation = ? AND fetched_date = ?
            ORDER BY status ASC
            """,
            (team_abbreviation.upper(), today),
        ).fetchall()
    finally:
        conn.close()

    return [
        {"player_id": r[0], "player_name": r[1], "status": r[2], "description": r[3]}
        for r in rows
    ]
