"""
Database initialization and connection utilities.
Creates all tables on first run.
"""

import sqlite3
from pathlib import Path
from config.settings import DB_PATH
from config.logging_config import setup_logging

logger = setup_logging("db")


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path = DB_PATH):
    """
    Initialize all database tables.
    Safe to call repeatedly — uses CREATE TABLE IF NOT EXISTS.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = """
    -- Raw game logs pulled from NBA API
    CREATE TABLE IF NOT EXISTS player_game_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        team_abbreviation TEXT,
        game_id TEXT NOT NULL,
        game_date DATE NOT NULL,
        season TEXT NOT NULL,
        matchup TEXT,
        is_home INTEGER,
        wl TEXT,
        minutes REAL,
        points INTEGER,
        fgm INTEGER,
        fga INTEGER,
        fg_pct REAL,
        fg3m INTEGER,
        fg3a INTEGER,
        ftm INTEGER,
        fta INTEGER,
        rebounds INTEGER,
        assists INTEGER,
        steals INTEGER,
        blocks INTEGER,
        turnovers INTEGER,
        plus_minus REAL,
        usage_rate REAL,
        UNIQUE(player_id, game_id)
    );

    -- Team-level stats for opponent defensive ratings
    CREATE TABLE IF NOT EXISTS team_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id INTEGER NOT NULL,
        team_abbreviation TEXT NOT NULL,
        season TEXT NOT NULL,
        game_date DATE NOT NULL,
        defensive_rating REAL,
        pace REAL,
        opp_points_allowed REAL,
        UNIQUE(team_id, game_date)
    );

    -- Sportsbook player prop lines
    CREATE TABLE IF NOT EXISTS sportsbook_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT NOT NULL,
        market TEXT NOT NULL,
        game_date DATE NOT NULL,
        bookmaker TEXT,
        line REAL,
        over_price REAL,
        under_price REAL,
        implied_over_prob REAL,
        implied_under_prob REAL,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_name, game_date, bookmaker)
    );

    -- Prediction market odds
    CREATE TABLE IF NOT EXISTS prediction_market_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        player_name TEXT,
        market_title TEXT,
        market_id TEXT,
        game_date DATE,
        line REAL,
        over_prob REAL,
        under_prob REAL,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Model predictions
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        game_date DATE NOT NULL,
        predicted_points REAL,
        predicted_over_prob REAL,
        sportsbook_line REAL,
        sportsbook_implied_prob REAL,
        kalshi_prob REAL,
        polymarket_prob REAL,
        model_type TEXT,
        edge REAL,
        confidence TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, game_date, model_type)
    );

    -- Actual results for scoring predictions
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        game_date DATE NOT NULL,
        actual_points INTEGER,
        sportsbook_line REAL,
        went_over INTEGER,
        prediction_id INTEGER REFERENCES predictions(id),
        model_was_correct INTEGER,
        UNIQUE(player_id, game_date)
    );

    -- Computed features cache
    CREATE TABLE IF NOT EXISTS player_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL,
        game_date DATE NOT NULL,
        features_json TEXT NOT NULL,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, game_date)
    );

    -- Model retraining history for tracking improvement over time
    CREATE TABLE IF NOT EXISTS learning_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        retrain_date DATE NOT NULL,
        scored_predictions INTEGER DEFAULT 0,
        model_name TEXT,
        mae REAL,
        rmse REAL,
        directional_accuracy REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Paper trading bet ledger
    CREATE TABLE IF NOT EXISTS paper_bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_date DATE NOT NULL,
        player_name TEXT NOT NULL,
        bet_direction TEXT NOT NULL,
        bet_amount REAL NOT NULL,
        odds_source TEXT NOT NULL,
        implied_prob REAL,
        model_prob REAL,
        edge REAL,
        kelly_fraction REAL,
        decimal_odds REAL,
        status TEXT DEFAULT 'pending',
        payout REAL,
        profit_loss REAL,
        settled_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Bankroll balance history
    CREATE TABLE IF NOT EXISTS bankroll_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_date DATE NOT NULL,
        action TEXT NOT NULL,
        amount REAL NOT NULL,
        balance_after REAL NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Injury and availability tracking
    CREATE TABLE IF NOT EXISTS injury_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL,
        player_name TEXT NOT NULL,
        team_id INTEGER,
        team_abbreviation TEXT,
        status TEXT NOT NULL,
        description TEXT,
        fetched_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(player_id, fetched_date)
    );
    """

    with get_connection(db_path) as conn:
        conn.executescript(schema)
        conn.commit()

    # Add RL columns to paper_bets if they don't exist yet (idempotent migrations)
    _migrate_paper_bets_rl(db_path)

    logger.info(f"Database initialized at {db_path}")


def _migrate_paper_bets_rl(db_path: str | Path):
    """Add rl_state and rl_action columns to paper_bets if missing."""
    with get_connection(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(paper_bets)").fetchall()}
        if "rl_state" not in existing:
            conn.execute("ALTER TABLE paper_bets ADD COLUMN rl_state TEXT")
        if "rl_action" not in existing:
            conn.execute("ALTER TABLE paper_bets ADD COLUMN rl_action TEXT")
        conn.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
