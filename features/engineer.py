"""
Feature engineering for the NBA points prediction model.
Transforms raw game logs into predictive features.

CRITICAL: All features must only use data BEFORE the target game_date.
No data leakage.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config.settings import DB_PATH, ROLLING_WINDOWS
from config.logging_config import setup_logging

logger = setup_logging("feature_engineer")


# ── helpers ──────────────────────────────────────────────────────────────────

def _scalar(v) -> float | int | None:
    """
    Force any value to a plain Python scalar (float/int/None).
    Prevents pandas 'All arrays must be of the same length' when building
    a DataFrame from a list of feature dicts that contain numpy scalars,
    0-d arrays, or accidentally-series values.
    """
    if v is None:
        return None
    if isinstance(v, (pd.Series, pd.DataFrame)):
        # Should never be a multi-element series; take first element
        if isinstance(v, pd.Series) and len(v) == 1:
            v = v.iloc[0]
        elif isinstance(v, pd.Series) and len(v) > 1:
            return float(v.mean())  # fallback: average of the series
        else:
            return None
    if isinstance(v, np.ndarray):
        if v.ndim == 0 or v.size == 1:
            v = v.flat[0]
        else:
            return None  # multi-element array — should not appear
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return int(v)
    return v


def _rolling_stat(series: pd.Series, window: int, stat: str = "mean") -> float:
    """Compute rolling stat over the last N games (tail). Returns NaN if insufficient data."""
    tail = series.iloc[-window:] if len(series) >= window else series
    if tail.empty:
        return np.nan
    if stat == "mean":
        return tail.mean()
    if stat == "median":
        return tail.median()
    if stat == "std":
        return tail.std(ddof=0) if len(tail) > 1 else 0.0
    if stat == "min":
        return tail.min()
    if stat == "max":
        return tail.max()
    return np.nan


def _linear_slope(series: pd.Series) -> float:
    """Compute linear regression slope (trend) over a series."""
    arr = series.dropna().values
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    try:
        slope, *_ = scipy_stats.linregress(x, arr)
        return float(slope)
    except Exception:
        return 0.0


def _load_player_logs(player_id: int, before_date: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all game logs for a player strictly before the given date."""
    df = pd.read_sql_query(
        """
        SELECT * FROM player_game_logs
        WHERE player_id = ? AND game_date < ?
        ORDER BY game_date ASC
        """,
        conn,
        params=(player_id, before_date),
    )
    return df


def _load_injury_data(game_date: str, conn: sqlite3.Connection) -> dict[int, dict]:
    """
    Load injury_status rows for game_date from the DB.
    Returns {player_id: {player_name, team_abbreviation, status}} for OUT/QUESTIONABLE players.
    Falls back to {} if the table is missing or empty.
    """
    try:
        rows = conn.execute(
            """
            SELECT player_id, player_name, team_abbreviation, status
            FROM injury_status
            WHERE fetched_date = ?
              AND UPPER(status) IN ('OUT', 'DOUBTFUL', 'QUESTIONABLE', 'DAY-TO-DAY')
            """,
            (game_date,),
        ).fetchall()
    except Exception:
        return {}
    return {
        row[0]: {"player_name": row[1], "team_abbreviation": row[2], "status": row[3]}
        for row in rows
    }


def _get_player_season_avg(player_id: int, conn: sqlite3.Connection, window: int = 82) -> float:
    """Return rolling season PPG for a player (last `window` games)."""
    try:
        rows = conn.execute(
            """
            SELECT points FROM player_game_logs
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
            """,
            (player_id, window),
        ).fetchall()
    except Exception:
        return 0.0
    if not rows:
        return 0.0
    return float(np.mean([r[0] for r in rows if r[0] is not None]))


def _load_team_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the most recent team stats available."""
    df = pd.read_sql_query(
        """
        SELECT team_abbreviation, defensive_rating, pace, opp_points_allowed
        FROM team_stats
        ORDER BY game_date DESC
        """,
        conn,
    )
    # Keep only most recent row per team
    return df.groupby("team_abbreviation").first().reset_index()


STAR_PPG_THRESHOLD = 18.0  # PPG threshold to classify a player as a "star"


def _build_injury_features(
    player_id: int,
    game_date: str,
    logs: pd.DataFrame,
    injury_map: dict[int, dict],
    conn: sqlite3.Connection,
) -> dict:
    """
    Build 3 injury-context features.

    - key_teammates_out: count of OUT teammates averaging ≥ STAR_PPG_THRESHOLD
    - star_pts_missing: sum of season PPG for those OUT star teammates
      (high value = more usage redistributed to this player)
    - opp_star_injured: 1 if any opponent star (≥ threshold) is OUT (easier matchup)
    """
    base = {"key_teammates_out": 0, "star_pts_missing": 0.0, "opp_star_injured": 0}

    if logs.empty or not injury_map:
        return base

    # Determine player's current team from most recent log
    my_team = str(logs["team_abbreviation"].values[-1]).upper() if "team_abbreviation" in logs.columns else ""
    if not my_team:
        return base

    # Determine opponent from matchup
    matchup = str(logs["matchup"].values[-1]) if "matchup" in logs.columns else ""
    opp_team = _parse_opponent(matchup, my_team)

    key_out = 0
    star_pts = 0.0
    opp_star_out = 0

    for pid, info in injury_map.items():
        if pid == player_id:
            continue  # skip self (own injury handled upstream by not predicting)

        team = str(info.get("team_abbreviation", "")).upper()
        ppg = _get_player_season_avg(pid, conn)

        if ppg < STAR_PPG_THRESHOLD:
            continue  # not a star player

        if team == my_team:
            key_out += 1
            star_pts += ppg
        elif team == opp_team:
            opp_star_out = 1

    return {
        "key_teammates_out": key_out,
        "star_pts_missing": round(star_pts, 1),
        "opp_star_injured": opp_star_out,
    }


# ── per-player feature computation ───────────────────────────────────────────

def build_features_for_player(
    player_id: int,
    game_date: str,
    opponent_abbr: str = "",
    is_home: int = 0,
    db_path: str | Path = DB_PATH,
) -> dict:
    """
    Compute all features for player going into a given game.
    Uses only data BEFORE game_date. Returns flat feature dict.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        logs = _load_player_logs(player_id, game_date, conn)
        team_stats = _load_team_stats(conn)
        injury_map = _load_injury_data(game_date, conn)

        # Build injury features using open connection (needs PPG lookups)
        injury_features = _build_injury_features(
            player_id, game_date, logs, injury_map, conn
        )
    finally:
        conn.close()

    features: dict = {}

    if logs.empty:
        return features

    pts = logs["points"].astype(float)
    fga = logs["fga"].astype(float)
    fg_pct = logs["fg_pct"].astype(float)
    fg3a = logs["fg3a"].astype(float)
    fg3m = logs["fg3m"].astype(float)
    fta = logs["fta"].astype(float)
    ftm = logs["ftm"].astype(float)
    mins = logs["minutes"].astype(float)
    # fgm: use stored value if available, fall back to fga * fg_pct
    fgm = logs["fgm"].fillna(logs["fga"] * logs["fg_pct"]).astype(float)
    plus_minus = logs["plus_minus"].astype(float)

    # ── Scoring features ──────────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        features[f"avg_points_last_{w}"] = _scalar(_rolling_stat(pts, w, "mean"))
    # Use last 82 games as "season average" — one NBA season — so multi-season
    # DB data doesn't drag the average down with stale prior-season numbers.
    features["avg_points_season"] = _scalar(_rolling_stat(pts, 82, "mean"))
    features["median_points_last_10"] = _scalar(_rolling_stat(pts, 10, "median"))
    features["std_points_last_10"] = _scalar(_rolling_stat(pts, 10, "std"))
    features["min_points_last_10"] = _scalar(_rolling_stat(pts, 10, "min"))
    features["max_points_last_10"] = _scalar(_rolling_stat(pts, 10, "max"))
    features["points_trend"] = _scalar(_linear_slope(pts.iloc[-10:] if len(pts) >= 10 else pts))

    # ── Shooting features ─────────────────────────────────────────────────
    features["avg_fga_last_5"] = _scalar(_rolling_stat(fga, 5, "mean"))
    features["avg_fg_pct_last_5"] = _scalar(_rolling_stat(fg_pct, 5, "mean"))
    features["avg_fg3a_last_5"] = _scalar(_rolling_stat(fg3a, 5, "mean"))
    fg3_pct = (fg3m / fg3a.replace(0, np.nan)).fillna(0)
    features["avg_fg3_pct_last_5"] = _scalar(_rolling_stat(fg3_pct, 5, "mean"))
    features["avg_fta_last_5"] = _scalar(_rolling_stat(fta, 5, "mean"))
    ft_pct = (ftm / fta.replace(0, np.nan)).fillna(0)
    features["avg_ft_pct_season"] = _scalar(ft_pct.mean())

    # ── Efficiency features ────────────────────────────────────────────────
    # Use 5-game window totals to avoid per-game division-by-zero issues.
    _w = min(5, len(logs))
    _pts5  = pts.iloc[-_w:].sum()
    _fga5  = fga.iloc[-_w:].sum()
    _fta5  = fta.iloc[-_w:].sum()
    _fgm5  = fgm.iloc[-_w:].sum()
    _fg3m5 = fg3m.iloc[-_w:].sum()
    _fg3a5 = fg3a.iloc[-_w:].sum()

    # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
    _ts_denom = 2.0 * (_fga5 + 0.44 * _fta5)
    features["ts_pct_last5"] = _scalar(_pts5 / _ts_denom if _ts_denom > 0 else np.nan)

    # Effective FG%: (FGM + 0.5 * FG3M) / FGA
    features["efg_pct_last5"] = _scalar(
        (_fgm5 + 0.5 * _fg3m5) / _fga5 if _fga5 > 0 else np.nan
    )

    # Free throw rate (FTA / FGA): measures ability to draw fouls
    features["ft_rate_last5"] = _scalar(_fta5 / _fga5 if _fga5 > 0 else np.nan)

    # Three-point attempt rate (3PA / FGA): dependence on 3s
    features["fg3_rate_last5"] = _scalar(_fg3a5 / _fga5 if _fga5 > 0 else np.nan)

    # Points per minute (last 5 games individual rates averaged)
    pts_per_min_series = (pts / mins.replace(0, np.nan)).dropna()
    features["pts_per_min_last5"] = _scalar(_rolling_stat(pts_per_min_series, 5, "mean"))

    # FG% slope over last 10 (shooting confidence trend: positive = heating up)
    features["fg_pct_trend"] = _scalar(
        _linear_slope(fg_pct.iloc[-10:] if len(fg_pct) >= 10 else fg_pct)
    )

    # ── Form / Streak features ─────────────────────────────────────────────
    # Hot/cold ratio: last-3 avg relative to last-20 avg
    # >1.10 = hot streak, <0.90 = cold streak, ~1.0 = normal form
    _avg3  = _rolling_stat(pts, 3, "mean")
    _avg20 = _rolling_stat(pts, 20, "mean")
    features["hot_cold_ratio"] = _scalar(
        _avg3 / _avg20 if (not np.isnan(_avg20) and _avg20 > 0) else 1.0
    )

    # Relative form change: how much has avg_last5 moved vs avg_last15?
    _avg15 = _rolling_stat(pts, 15, "mean")
    _avg5_v = features.get("avg_points_last_5", np.nan)
    if (not np.isnan(_avg15) and _avg15 > 0
            and isinstance(_avg5_v, (int, float)) and not (isinstance(_avg5_v, float) and np.isnan(_avg5_v))):
        features["form_change_pct"] = float((_avg5_v - _avg15) / _avg15)
    else:
        features["form_change_pct"] = 0.0

    # ── Minutes & Usage features ──────────────────────────────────────────
    features["avg_minutes_last_5"] = _scalar(_rolling_stat(mins, 5, "mean"))
    features["avg_minutes_last_10"] = _scalar(_rolling_stat(mins, 10, "mean"))
    features["minutes_trend"] = _scalar(_linear_slope(mins.iloc[-10:] if len(mins) >= 10 else mins))

    # Minutes variance: high std = uncertain playing time (injury/foul trouble signal)
    features["min_variance_last10"] = _scalar(_rolling_stat(mins, 10, "std"))

    # Minutes floor: minimum minutes in last 5 games (guaranteed playing time)
    features["min_floor_last5"] = _scalar(_rolling_stat(mins, 5, "min"))

    # Usage rate: pull from logs if available, else NaN
    if "usage_rate" in logs.columns and logs["usage_rate"].notna().any():
        usage = logs["usage_rate"].astype(float)
        # Use .values[-1] instead of .iloc[-1] to guarantee a numpy scalar, not a Series
        features["usage_rate"] = _scalar(usage.values[-1]) if not usage.empty else np.nan
        features["avg_usage_last_5"] = _scalar(_rolling_stat(usage, 5, "mean"))
    else:
        features["usage_rate"] = np.nan
        features["avg_usage_last_5"] = np.nan

    # ── Context features ──────────────────────────────────────────────────
    features["is_home"] = int(is_home)

    home_games = logs[logs["is_home"] == 1]["points"].astype(float)
    away_games = logs[logs["is_home"] == 0]["points"].astype(float)
    features["home_away_avg_diff"] = _scalar(
        (home_games.mean() - away_games.mean())
        if len(home_games) > 0 and len(away_games) > 0
        else 0.0
    )

    # Rest days
    if len(logs) >= 1:
        last_game_date = pd.to_datetime(logs["game_date"].values[-1])
        target_dt = pd.to_datetime(game_date)
        rest_days = int((target_dt - last_game_date).days) - 1
        features["rest_days"] = max(0, rest_days)
        features["is_back_to_back"] = 1 if features["rest_days"] == 0 else 0
    else:
        features["rest_days"] = 3
        features["is_back_to_back"] = 0

    # Games in last 7 days
    seven_days_ago = pd.to_datetime(game_date) - pd.Timedelta(days=7)
    recent_games = logs[pd.to_datetime(logs["game_date"]) >= seven_days_ago]
    features["games_played_last_7_days"] = int(len(recent_games))

    # Back-to-back road game: extra fatigue (away B2B harder than home B2B)
    features["b2b_road"] = int(features.get("is_back_to_back", 0) == 1 and not is_home)

    # ── Schedule / Travel features ────────────────────────────────────────
    # Consecutive away games: road trip length (accumulating travel fatigue)
    is_home_vals = logs["is_home"].values
    consec_away = 0
    for _h in reversed(is_home_vals):
        if _h == 0:
            consec_away += 1
        else:
            break
    features["consecutive_away_games"] = int(consec_away)

    # Altitude disadvantage: visiting DEN or UTA incurs real ~2-3% scoring penalty
    _HIGH_ALT = {"DEN", "UTA"}
    features["altitude_disadvantage"] = int(not is_home and opponent_abbr.upper() in _HIGH_ALT)

    # ── Performance context features ──────────────────────────────────────
    # Average ±/- last 10 games: captures whether team is playing well (winning = more offense)
    features["avg_plus_minus_last10"] = _scalar(_rolling_stat(plus_minus, 10, "mean"))

    # Blowout game rate last 10: high rate = unpredictable minutes due to lopsided scores
    _pm10 = plus_minus.iloc[-10:] if len(plus_minus) >= 10 else plus_minus
    features["blowout_rate_last10"] = _scalar(
        (_pm10.abs() > 12).sum() / max(len(_pm10), 1)
    )

    # Foul drawing rate: FTA per minute (last 10 games)
    _fta_per_min = (fta / mins.replace(0, np.nan)).fillna(0)
    features["foul_drawing_rate_last10"] = _scalar(_rolling_stat(_fta_per_min, 10, "mean"))

    # ── Opponent features ─────────────────────────────────────────────────
    league_avg_def_rating = float(team_stats["defensive_rating"].mean()) if not team_stats.empty else 112.0
    league_avg_pace = float(team_stats["pace"].mean()) if not team_stats.empty else 100.0

    opp_row = team_stats[team_stats["team_abbreviation"] == opponent_abbr]
    if not opp_row.empty:
        opp_def_rating = float(opp_row["defensive_rating"].values[0])
        opp_pace = float(opp_row["pace"].values[0])
        opp_pts_allowed = float(opp_row["opp_points_allowed"].values[0]) if "opp_points_allowed" in opp_row.columns else np.nan
    else:
        opp_def_rating = league_avg_def_rating
        opp_pace = league_avg_pace
        opp_pts_allowed = np.nan

    features["opp_defensive_rating"] = opp_def_rating
    features["opp_pace"] = opp_pace
    features["pace_diff"] = float(opp_pace - league_avg_pace)
    features["opp_points_allowed"] = opp_pts_allowed

    # ── Derived / interaction features ────────────────────────────────────
    avg_last_5 = features.get("avg_points_last_5", np.nan)
    avg_season = features.get("avg_points_season", np.nan)
    avg_last_10 = features.get("avg_points_last_10", np.nan)
    std_last_10 = features.get("std_points_last_10", np.nan)

    # Guard against NaN in boolean checks with explicit isinstance tests
    def _safe_pos(v) -> bool:
        return isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) and v > 0

    features["recent_vs_season_ratio"] = float(
        avg_last_5 / avg_season if _safe_pos(avg_season) and _safe_pos(avg_last_5) else 1.0
    )
    features["consistency_score"] = float(
        1 - (std_last_10 / avg_last_10) if _safe_pos(avg_last_10) else 0.0
    )
    avg_fga_5 = features.get("avg_fga_last_5", np.nan)
    avg_fg_pct_5 = features.get("avg_fg_pct_last_5", np.nan)
    features["volume_efficiency_score"] = float(
        avg_fga_5 * avg_fg_pct_5
        if _safe_pos(avg_fga_5) and not (isinstance(avg_fg_pct_5, float) and np.isnan(avg_fg_pct_5))
        else 0.0
    )

    rest = features.get("rest_days", 2)
    rest_multiplier = 1.0 + 0.01 * min(rest, 5)
    features["rest_adjusted_avg"] = float(
        avg_last_5 * rest_multiplier
        if isinstance(avg_last_5, (int, float)) and not (isinstance(avg_last_5, float) and np.isnan(avg_last_5))
        else 0.0
    )

    features["matchup_adjusted_avg"] = float(
        avg_last_10 * (league_avg_def_rating / opp_def_rating)
        if _safe_pos(opp_def_rating) and _safe_pos(avg_last_10)
        else (avg_last_10 if _safe_pos(avg_last_10) else 0.0)
    )

    features["games_in_db"] = int(len(logs))

    # ── Injury / lineup context features ──────────────────────────────────
    features.update(injury_features)

    return features


# ── full training dataset builder ────────────────────────────────────────────

def build_training_dataset(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """
    For all players and games in the database, compute features and pair
    with actual points scored. Returns DataFrame ready for model training.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))

    # Get all distinct player-game combinations
    games_df = pd.read_sql_query(
        """
        SELECT player_id, player_name, game_date, points, is_home, matchup, team_abbreviation
        FROM player_game_logs
        ORDER BY game_date ASC
        """,
        conn,
    )
    conn.close()

    if games_df.empty:
        logger.warning("No game log data found — run backfill first")
        return pd.DataFrame()

    logger.info(f"Building features for {len(games_df)} player-game rows")
    rows = []

    for idx, row in games_df.iterrows():
        player_id = int(row["player_id"])
        game_date = row["game_date"]
        actual_points = row["points"]

        # Parse opponent abbreviation from matchup (e.g., "LAL @ GSW" or "LAL vs. GSW")
        matchup = str(row.get("matchup", ""))
        team_abbr = str(row.get("team_abbreviation", ""))
        opponent = _parse_opponent(matchup, team_abbr)

        is_home = int(row.get("is_home", 0))

        features = build_features_for_player(
            player_id=player_id,
            game_date=game_date,
            opponent_abbr=opponent,
            is_home=is_home,
            db_path=db_path,
        )

        if not features or features.get("games_in_db", 0) < 5:
            continue  # Not enough history

        features["player_id"] = player_id
        features["player_name"] = str(row["player_name"])
        features["game_date"] = str(game_date)
        features["target_points"] = float(actual_points)

        # Coerce all values to Python scalars — prevents pandas
        # "All arrays must be of the same length" during DataFrame construction
        rows.append({k: _scalar(v) if not isinstance(v, str) else v for k, v in features.items()})

        if (idx + 1) % 500 == 0:
            logger.info(f"Processed {idx + 1}/{len(games_df)} rows")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Training dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


def _parse_opponent(matchup: str, team_abbr: str) -> str:
    """Extract opponent team abbreviation from matchup string."""
    if not matchup:
        return ""
    # "LAL vs. GSW" → opponent is GSW when team is LAL
    # "LAL @ GSW"   → opponent is GSW when team is LAL
    parts = re.split(r"vs\.|@", matchup, flags=re.IGNORECASE)
    if len(parts) == 2:
        left = parts[0].strip().upper()
        right = parts[1].strip().upper()
        team_upper = team_abbr.upper()
        if left == team_upper:
            return right
        return left
    return ""


def get_feature_names(db_path: str | Path = DB_PATH) -> list[str]:
    """Return the list of feature column names (excluding metadata)."""
    exclude = {"player_id", "player_name", "game_date", "target_points", "games_in_db"}
    df = build_training_dataset(db_path)
    if df.empty:
        return []
    return [c for c in df.columns if c not in exclude]


def save_features_to_db(player_id: int, game_date: str, features: dict, db_path: str | Path = DB_PATH):
    """Cache computed features for a player-game in the DB."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_features (player_id, game_date, features_json)
            VALUES (?, ?, ?)
            """,
            (player_id, game_date, json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in features.items()})),
        )
        conn.commit()
    finally:
        conn.close()
