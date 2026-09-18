"""
Export a JSON snapshot of the predictor's state for the public dashboard.

The dashboard is hosted on GitHub Pages, which is static — it cannot reach the
FastAPI server or the SQLite file. So this script flattens everything the page
needs into one small JSON file that gets committed alongside it.

Usage:
    python -m scripts.export_dashboard                      # writes to ./dashboard_export.json
    python -m scripts.export_dashboard --out ../VedaantK.github.io/nba-predictor/data.json

Re-run it after a training run or a daily pipeline, then commit the result to
refresh the public page.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH, MODELS_DIR, NBA_SEASONS
from config.logging_config import setup_logging

logger = setup_logging("export_dashboard")

MODEL_ORDER = ["random_forest", "linear_regression", "xgboost", "neural_net"]


def _scalar(conn, sql, default=0):
    try:
        row = conn.execute(sql).fetchone()
        return row[0] if row and row[0] is not None else default
    except sqlite3.Error:
        return default


def collect_dataset(conn) -> dict:
    """Shape and coverage of the training data actually in the database."""
    seasons = [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM player_game_logs WHERE season != '' ORDER BY season"
    ).fetchall()]

    per_season = [
        {"season": s, "rows": n}
        for s, n in conn.execute(
            "SELECT season, COUNT(*) FROM player_game_logs "
            "WHERE season != '' GROUP BY season ORDER BY season"
        ).fetchall()
    ]

    return {
        "players": _scalar(conn, "SELECT COUNT(DISTINCT player_id) FROM player_game_logs"),
        "game_rows": _scalar(conn, "SELECT COUNT(*) FROM player_game_logs"),
        "seasons": seasons,
        "seasons_configured": len(NBA_SEASONS),
        "per_season": per_season,
        "first_game": _scalar(conn, "SELECT MIN(game_date) FROM player_game_logs", None),
        "last_game": _scalar(conn, "SELECT MAX(game_date) FROM player_game_logs", None),
        "team_stat_rows": _scalar(conn, "SELECT COUNT(*) FROM team_stats"),
        "team_snapshots": _scalar(conn, "SELECT COUNT(DISTINCT game_date) FROM team_stats"),
    }


def collect_models() -> dict:
    """Training metrics, written by models/train.py."""
    path = Path(MODELS_DIR) / "metrics.json"
    if not path.exists():
        return {"trained": False, "models": [], "note": "No metrics.json — run train.py"}

    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read metrics.json: {e}")
        return {"trained": False, "models": []}

    scored = blob.get("models", {})
    attempted = blob.get("attempted", list(scored.keys()))

    models = []
    for name in MODEL_ORDER:
        if name not in attempted:
            continue
        m = scored.get(name)
        models.append({
            "name": name,
            "trained": m is not None,
            "mae": m.get("mae") if m else None,
            "rmse": m.get("rmse") if m else None,
            # Deliberately NOT called accuracy: _directional_accuracy is
            # evaluated against the median of the test set, not a sportsbook
            # line, so it measures above/below median — not an edge.
            "median_split_accuracy": m.get("directional_accuracy") if m else None,
        })

    return {
        "trained": bool(scored),
        "trained_at": blob.get("trained_at"),
        "train_rows": blob.get("train_rows"),
        "test_rows": blob.get("test_rows"),
        "n_features": blob.get("n_features"),
        "models": models,
    }


def collect_feature_importance(top_n: int = 12) -> list:
    """Top features from whichever tree model is available."""
    for name in ("random_forest", "xgboost"):
        path = Path(MODELS_DIR) / f"{name}_importance.json"
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rows = [r for r in rows if r.get("importance") is not None]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        total = sum(r["importance"] for r in rows) or 1.0
        return [
            {"feature": r["feature"],
             "importance": round(r["importance"] / total, 4),
             "source": name}
            for r in rows[:top_n]
        ]
    return []


def collect_predictions(conn) -> dict:
    """Prediction and scoring activity, if any has happened yet."""
    total = _scalar(conn, "SELECT COUNT(*) FROM predictions")
    scored = _scalar(conn, "SELECT COUNT(*) FROM results WHERE model_was_correct IS NOT NULL")
    correct = _scalar(conn, "SELECT COUNT(*) FROM results WHERE model_was_correct = 1")
    return {
        "total": total,
        "scored": scored,
        "correct": correct,
        "hit_rate": round(correct / scored, 4) if scored else None,
        "latest_date": _scalar(conn, "SELECT MAX(game_date) FROM predictions", None),
    }


def collect_sources(conn) -> list:
    """
    Which feeds have actually delivered rows into the database.

    Status here reflects stored data, not a live probe — the page is static and
    must not imply a health check it did not perform.
    """
    sb = _scalar(conn, "SELECT COUNT(*) FROM sportsbook_lines")
    kalshi = _scalar(conn, "SELECT COUNT(*) FROM prediction_market_lines WHERE source='kalshi'")
    poly = _scalar(conn, "SELECT COUNT(*) FROM prediction_market_lines WHERE source='polymarket'")
    injuries = _scalar(conn, "SELECT COUNT(*) FROM injury_status")
    logs = _scalar(conn, "SELECT COUNT(*) FROM player_game_logs")

    return [
        {"name": "NBA Stats", "detail": "player game logs, team ratings",
         "rows": logs, "status": "live" if logs else "empty"},
        {"name": "The Odds API", "detail": "sportsbook player props",
         "rows": sb, "status": "live" if sb else "awaiting season"},
        {"name": "Kalshi", "detail": "prediction market",
         "rows": kalshi, "status": "live" if kalshi else "awaiting fix"},
        {"name": "Polymarket", "detail": "prediction market",
         "rows": poly, "status": "live" if poly else "awaiting fix"},
        {"name": "ESPN", "detail": "injury reports",
         "rows": injuries, "status": "live" if injuries else "awaiting fix"},
    ]


def build_export(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        export = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": collect_dataset(conn),
            "models": collect_models(),
            "feature_importance": collect_feature_importance(),
            "predictions": collect_predictions(conn),
            "sources": collect_sources(conn),
        }
    finally:
        conn.close()
    return export


def main():
    ap = argparse.ArgumentParser(description="Export dashboard JSON")
    ap.add_argument("--out", default="dashboard_export.json", help="output path")
    ap.add_argument("--db", default=str(DB_PATH), help="database path")
    args = ap.parse_args()

    export = build_export(Path(args.db))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, indent=2))

    ds, md = export["dataset"], export["models"]
    logger.info(f"Wrote {out}")
    logger.info(f"  {ds['players']} players · {ds['game_rows']:,} game rows · {len(ds['seasons'])} seasons")
    logger.info(f"  {sum(1 for m in md['models'] if m['trained'])}/{len(md['models'])} models trained")


if __name__ == "__main__":
    main()
