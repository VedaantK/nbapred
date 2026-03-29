"""
FastAPI backend for the NBA predictor dashboard.
Run with: uvicorn api.server:app --reload --port 8000
"""

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from config.settings import DB_PATH, MODELS_DIR
from config.db import init_db
from config.logging_config import setup_logging
from models.evaluate import (
    get_model_performance, get_calibration_data, get_market_comparison,
    score_past_predictions, analyze_prediction_errors, get_learning_history,
    compute_model_weights,
)
from models.train import save_model_weights

try:
    from models.paper_trader import (
        generate_paper_bets, settle_paper_bets, get_bankroll,
        reset_bankroll, get_trader_performance,
    )
except ImportError as _pt_err:
    generate_paper_bets = None  # type: ignore
    settle_paper_bets = None    # type: ignore
    get_bankroll = None         # type: ignore
    reset_bankroll = None       # type: ignore
    get_trader_performance = None  # type: ignore

# Pre-import all pipeline dependencies so they're loaded at server startup,
# not on first button click (avoids a 15-30s freeze at the "starting" step).
try:
    from ingestion.odds_api import fetch_and_store_all_props
    from ingestion.kalshi import fetch_and_store_kalshi_markets
    from ingestion.polymarket import fetch_and_store_polymarket_markets
    from ingestion.injuries import fetch_and_store_injuries, get_team_injury_report
    from models.predict import predict_today
    from models.rl_agent import RLBettingAgent
except ImportError as _import_err:
    fetch_and_store_all_props = None        # type: ignore
    fetch_and_store_kalshi_markets = None   # type: ignore
    fetch_and_store_polymarket_markets = None  # type: ignore
    fetch_and_store_injuries = None         # type: ignore
    get_team_injury_report = None           # type: ignore
    predict_today = None                    # type: ignore
    RLBettingAgent = None                   # type: ignore
    import warnings
    warnings.warn(f"Pipeline imports failed: {_import_err}")

logger = setup_logging("api")

app = FastAPI(title="NBA Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db(DB_PATH)
    logger.info("API server started")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ── Predictions endpoints ─────────────────────────────────────────────────────

@app.get("/api/predictions/today")
async def get_today_predictions(game_date: str | None = None):
    """
    Return predictions for a given date sorted by edge descending.
    Defaults to today. Pass ?game_date=YYYY-MM-DD to query another day.
    """
    today = game_date or date.today().strftime("%Y-%m-%d")
    conn = _get_conn()

    rows = conn.execute(
        """
        SELECT
            p.player_name,
            p.game_date,
            AVG(p.predicted_points) AS predicted_points,
            AVG(p.predicted_over_prob) AS predicted_over_prob,
            p.sportsbook_line,
            p.sportsbook_implied_prob,
            p.kalshi_prob,
            p.polymarket_prob,
            AVG(p.edge) AS edge,
            p.confidence,
            GROUP_CONCAT(p.model_type || ':' || ROUND(p.predicted_over_prob, 3)) AS model_breakdown
        FROM predictions p
        WHERE p.game_date = ?
        GROUP BY p.player_name, p.game_date
        ORDER BY ABS(p.edge) DESC
        """,
        (today,),
    ).fetchall()

    conn.close()

    result = []
    for row in rows:
        d = _row_to_dict(row)
        # Parse model breakdown string
        if d.get("model_breakdown"):
            breakdown = {}
            for part in d["model_breakdown"].split(","):
                if ":" in part:
                    model, prob = part.split(":", 1)
                    try:
                        breakdown[model] = float(prob)
                    except ValueError:
                        pass
            d["model_breakdown"] = breakdown
        result.append(d)

    return {
        "date": today,
        "count": len(result),
        "predictions": result,
    }


@app.get("/api/predictions/history")
async def get_prediction_history(days: int = 30):
    """Return past predictions with outcomes."""
    conn = _get_conn()
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT
            p.player_name,
            p.game_date,
            AVG(p.predicted_points) AS predicted_points,
            p.sportsbook_line,
            AVG(p.predicted_over_prob) AS predicted_over_prob,
            p.confidence,
            AVG(p.edge) AS edge,
            r.actual_points,
            r.went_over,
            CASE WHEN (AVG(p.predicted_over_prob) > 0.5) = r.went_over THEN 1 ELSE 0 END AS correct
        FROM predictions p
        LEFT JOIN results r ON p.player_id = r.player_id AND p.game_date = r.game_date
        WHERE p.game_date >= ?
        GROUP BY p.player_name, p.game_date
        ORDER BY p.game_date DESC, ABS(p.edge) DESC
        """,
        (cutoff,),
    ).fetchall()

    conn.close()
    return {"days": days, "predictions": [_row_to_dict(r) for r in rows]}


# ── Performance endpoints ─────────────────────────────────────────────────────

@app.get("/api/performance/summary")
async def get_performance_summary(days: int = 30):
    """Return overall model performance metrics."""
    return get_model_performance(DB_PATH, days)


@app.get("/api/performance/calibration")
async def get_calibration():
    """Return calibration data for chart."""
    df = get_calibration_data(DB_PATH)
    if df.empty:
        return {"buckets": []}
    return {"buckets": df.to_dict(orient="records")}


@app.get("/api/performance/comparison")
async def get_comparison(days: int = 30):
    """Compare model vs sportsbook vs prediction market accuracy."""
    return get_market_comparison(DB_PATH, days)


# ── Feature importance endpoint ────────────────────────────────────────────────

@app.get("/api/features/importance")
async def get_feature_importance(model: str = "xgboost"):
    """Return ranked feature importance from a trained model."""
    from config.settings import MODELS_DIR
    models_dir = Path(MODELS_DIR)

    importance_path = models_dir / f"{model}_importance.json"
    if not importance_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model}' not trained yet")

    with open(importance_path) as f:
        data = json.load(f)

    return {"model": model, "importance": data[:30]}  # Top 30 features


# ── Scoring / learning state ───────────────────────────────────────────────────

_scoring_state: dict = {
    "status": "idle",   # idle | running | done | error
    "message": "",
    "scored": 0,
    "started_at": None,
    "finished_at": None,
}


@app.get("/api/scoring/status")
async def get_scoring_status():
    """Return current scoring/learning run status."""
    return dict(_scoring_state)


@app.post("/api/scoring/run")
async def trigger_scoring(background_tasks: BackgroundTasks, game_date: str | None = None):
    """Trigger scoring + error analysis + model retrain for a given date."""
    if _scoring_state["status"] == "running":
        return {"status": "already_running", "message": "Scoring is already running"}

    target_date = game_date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    def run_scoring():
        _scoring_state.update({
            "status": "running",
            "message": f"Scoring predictions for {target_date}...",
            "scored": 0,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        })
        try:
            # Step 1: score predictions against actual results
            result = _run_step("score", score_past_predictions, DB_PATH, target_date, timeout=120) or {}
            scored = result.get("scored", 0)
            _scoring_state["scored"] = scored
            _scoring_state["message"] = f"Scored {scored} predictions. Updating ensemble weights..."

            # Step 2: update dynamic ensemble weights (fast — reads learning_log, no feature rebuild)
            try:
                weights = compute_model_weights(DB_PATH)
                if weights:
                    save_model_weights(weights)
                    logger.info(f"Ensemble weights updated: {weights}")
            except Exception as _we:
                logger.warning(f"Weight update failed: {_we}")

            # Step 3: record in learning_log (weights-only entry)
            import sqlite3 as _sqlite3
            try:
                conn = _sqlite3.connect(str(DB_PATH))
                conn.execute("PRAGMA busy_timeout = 10000")
                conn.execute(
                    """
                    INSERT INTO learning_log
                    (retrain_date, scored_predictions, model_name, mae, rmse, directional_accuracy, notes)
                    VALUES (?, ?, 'ensemble', NULL, NULL, NULL, 'weights_updated')
                    """,
                    (target_date, scored),
                )
                conn.commit()
                conn.close()
            except Exception as _le:
                logger.debug(f"Learning log insert failed: {_le}")

            _scoring_state.update({
                "status": "done",
                "message": f"Scored {scored} predictions. Weights updated.",
                "finished_at": datetime.now().isoformat(),
            })
            logger.info(f"Daily scoring complete for {target_date} — scored {scored}")

        except Exception as e:
            _scoring_state.update({
                "status": "error",
                "message": str(e),
                "finished_at": datetime.now().isoformat(),
            })
            logger.error(f"Scoring run failed: {e}", exc_info=True)

    background_tasks.add_task(run_scoring)
    return {"status": "started", "game_date": target_date, "timestamp": datetime.now().isoformat()}


_retrain_state: dict = {
    "status": "idle",
    "message": "",
    "started_at": None,
    "finished_at": None,
}


@app.get("/api/training/status")
async def get_training_status():
    """Return status of a full model retrain run."""
    return dict(_retrain_state)


@app.post("/api/training/run")
async def trigger_full_retrain(background_tasks: BackgroundTasks):
    """
    Full model retrain — rebuilds features + retrains all 4 models from scratch.
    This is slow (5-30 min). Use /api/scoring/run for the fast daily update.
    """
    if _retrain_state["status"] == "running":
        return {"status": "already_running"}

    def run_retrain():
        _retrain_state.update({
            "status": "running",
            "message": "Building training dataset and retraining all models...",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        })
        try:
            from models.train import train_all_models
            train_results = _run_step("full_retrain", train_all_models, DB_PATH, timeout=1800) or {}

            if train_results:
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(str(DB_PATH))
                conn.execute("PRAGMA busy_timeout = 10000")
                today_str = date.today().strftime("%Y-%m-%d")
                try:
                    for name, info in train_results.items():
                        m = info["metrics"]
                        conn.execute(
                            """
                            INSERT INTO learning_log
                            (retrain_date, scored_predictions, model_name, mae, rmse,
                             directional_accuracy, notes)
                            VALUES (?, 0, ?, ?, ?, ?, 'full_retrain')
                            """,
                            (today_str, name, m["mae"], m["rmse"], m["directional_accuracy"]),
                        )
                    conn.commit()
                finally:
                    conn.close()

                weights = compute_model_weights(DB_PATH)
                if weights:
                    save_model_weights(weights)

            model_count = len(train_results) if train_results else 0
            _retrain_state.update({
                "status": "done" if model_count > 0 else "error",
                "message": f"Retrained {model_count} models." if model_count > 0 else "Retrain timed out or failed — check server logs.",
                "finished_at": datetime.now().isoformat(),
            })
        except Exception as e:
            _retrain_state.update({
                "status": "error",
                "message": str(e),
                "finished_at": datetime.now().isoformat(),
            })
            logger.error(f"Full retrain failed: {e}", exc_info=True)

    background_tasks.add_task(run_retrain)
    return {"status": "started", "timestamp": datetime.now().isoformat()}


@app.get("/api/performance/error-analysis")
async def get_error_analysis(game_date: str | None = None):
    """Return prediction error analysis for a given date."""
    return analyze_prediction_errors(DB_PATH, game_date)


@app.get("/api/performance/learning-history")
async def get_learning_history_endpoint():
    """Return model improvement history from learning_log."""
    return {"history": get_learning_history(DB_PATH)}


# ── Paper trader endpoints ─────────────────────────────────────────────────────

@app.get("/api/trader/status")
async def get_trader_status():
    """Return current bankroll and pending bet counts."""
    if get_bankroll is None:
        raise HTTPException(status_code=503, detail="Paper trader module not available")
    conn = _get_conn()
    try:
        pending = conn.execute(
            "SELECT COUNT(*) FROM paper_bets WHERE status = 'pending'"
        ).fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM paper_bets WHERE game_date = ?",
            (date.today().strftime("%Y-%m-%d"),),
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "current_balance": get_bankroll(DB_PATH),
        "pending_bets_count": pending,
        "today_bets_count": today,
    }


@app.get("/api/trader/bets")
async def get_trader_bets(game_date: str | None = None):
    """Return paper bets for a given date."""
    target = game_date or date.today().strftime("%Y-%m-%d")
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT * FROM paper_bets WHERE game_date = ?
            ORDER BY ABS(edge) DESC
            """,
            (target,),
        ).fetchall()
    finally:
        conn.close()
    return {"game_date": target, "bets": [_row_to_dict(r) for r in rows]}


@app.post("/api/trader/bets/generate")
async def trigger_generate_bets(game_date: str | None = None):
    """Generate paper bet recommendations for a date using existing predictions."""
    if generate_paper_bets is None:
        raise HTTPException(status_code=503, detail="Paper trader module not available")
    target = game_date or date.today().strftime("%Y-%m-%d")
    bets = generate_paper_bets(target, DB_PATH)
    return {"game_date": target, "generated": len(bets), "bets": bets}


@app.post("/api/trader/bets/settle")
async def trigger_settle_bets(game_date: str | None = None):
    """Settle pending paper bets for a date against actual results."""
    if settle_paper_bets is None:
        raise HTTPException(status_code=503, detail="Paper trader module not available")
    target = game_date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    result = settle_paper_bets(target, DB_PATH)
    return result


@app.post("/api/trader/bankroll/reset")
async def trigger_reset_bankroll(amount: float = 30.0):
    """Reset bankroll to a specified starting amount."""
    if reset_bankroll is None:
        raise HTTPException(status_code=503, detail="Paper trader module not available")
    reset_bankroll(amount, DB_PATH)
    return {"reset": True, "new_balance": amount}


@app.get("/api/trader/performance")
async def get_trader_performance_endpoint(days: int = 30):
    """Return cumulative P&L and performance breakdown for the paper trader."""
    if get_trader_performance is None:
        raise HTTPException(status_code=503, detail="Paper trader module not available")
    return get_trader_performance(DB_PATH, days)


# ── Shared background step runner ─────────────────────────────────────────────

def _run_step(name: str, func, *args, timeout: int):
    """
    Run func(*args) in a background thread with a hard timeout.
    Returns the result on success, None on timeout or error.
    Never raises — each step is fault-tolerant so callers always complete.
    """
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.warning(f"[{name}] timed out after {timeout}s — moving on")
        return None
    if exc[0] is not None:
        logger.warning(f"[{name}] failed: {exc[0]} — moving on")
        return None
    return result[0]


# ── Pipeline status tracking ───────────────────────────────────────────────────

_pipeline_state: dict = {
    "status": "idle",      # idle | running | done | error
    "step": "",
    "message": "",
    "predictions_count": 0,
    "started_at": None,
    "finished_at": None,
}


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Return current pipeline status for frontend polling."""
    return dict(_pipeline_state)


# ── Pipeline trigger endpoint ──────────────────────────────────────────────────

@app.post("/api/pipeline/run")
async def trigger_daily_pipeline(background_tasks: BackgroundTasks, game_date: str | None = None):
    """Manually trigger the daily pipeline for the given date (default today)."""
    target_date = game_date or date.today().strftime("%Y-%m-%d")
    if _pipeline_state["status"] == "running":
        return {"status": "already_running", "message": "Pipeline is already running"}

    # Check for import failures at startup
    if predict_today is None:
        return {"status": "error", "message": "Pipeline modules failed to import. Check server logs."}

    # Check for trained models NOW (fast, synchronous) so we can return an
    # immediate clear error instead of hanging at "starting" in the background.
    models_dir = Path(MODELS_DIR)
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        err = (
            "No trained models found. "
            "Run 'python -m scripts.backfill_data' first, then 'python train.py'."
        )
        _pipeline_state.update({
            "status": "error", "step": "error", "message": err,
            "finished_at": datetime.now().isoformat(),
        })
        return {"status": "error", "message": err}

    def run_pipeline():
        def _set(step: str, message: str = ""):
            _pipeline_state["step"] = step
            _pipeline_state["message"] = message
            logger.info(f"Pipeline [{step}]" + (f": {message}" if message else ""))

        _pipeline_state.update({
            "status": "running",
            "step": "sportsbook",
            "message": "Fetching sportsbook lines...",
            "predictions_count": 0,
            "game_date": target_date,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        })
        try:
            # Each step is non-fatal: timeout or error → log warning and continue.
            # The pipeline always reaches "done" or the final except.
            _set("injuries", f"Fetching injury reports for {target_date}...")
            if fetch_and_store_injuries is not None:
                _run_step("injuries", fetch_and_store_injuries, DB_PATH, target_date, timeout=60)

            _set("sportsbook", f"Fetching sportsbook lines for {target_date}...")
            _run_step("sportsbook", fetch_and_store_all_props, DB_PATH, target_date, timeout=45)

            _set("kalshi", "Fetching Kalshi prediction market odds...")
            _run_step("kalshi", fetch_and_store_kalshi_markets, DB_PATH, target_date, timeout=20)

            _set("polymarket", "Fetching Polymarket odds...")
            _run_step("polymarket", fetch_and_store_polymarket_markets, DB_PATH, target_date, timeout=20)

            _set("predicting", "Generating predictions...")
            predictions = _run_step("predict", predict_today, DB_PATH, target_date, timeout=120) or []

            _pipeline_state.update({
                "status": "done",
                "step": "done",
                "message": f"Generated {len(predictions)} predictions.",
                "predictions_count": len(predictions),
                "finished_at": datetime.now().isoformat(),
            })
            logger.info(f"Pipeline complete — {len(predictions)} predictions generated.")

        except Exception as e:
            _pipeline_state.update({
                "status": "error",
                "step": "error",
                "message": str(e),
                "finished_at": datetime.now().isoformat(),
            })
            logger.error(f"Pipeline failed: {e}", exc_info=True)

    background_tasks.add_task(run_pipeline)
    return {"status": "started", "timestamp": datetime.now().isoformat()}


# ── Daily results endpoints ──────────────────────────────────────────────────

@app.get("/api/results/daily")
async def get_daily_results(game_date: str | None = None):
    """
    Return per-player prediction vs actual comparison for a given date.
    Returns fetched=False when no results exist yet for that date.
    """
    target = game_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                p.player_name,
                p.sportsbook_line,
                ROUND(AVG(p.predicted_points), 1)  AS predicted_points,
                ROUND(AVG(p.predicted_over_prob), 4) AS predicted_over_prob,
                p.confidence,
                ROUND(AVG(p.edge), 4)               AS edge,
                r.actual_points,
                r.went_over,
                CASE WHEN (AVG(p.predicted_over_prob) > 0.5) = r.went_over
                     THEN 1 ELSE 0 END              AS correct
            FROM predictions p
            LEFT JOIN results r
                   ON p.player_id = r.player_id AND p.game_date = r.game_date
            WHERE p.game_date = ?
            GROUP BY p.player_name, p.game_date
            ORDER BY ABS(COALESCE(r.actual_points, 0) - AVG(p.predicted_points)) DESC
            """,
            (target,),
        ).fetchall()
    finally:
        conn.close()

    players = [_row_to_dict(r) for r in rows]

    # Check whether results have actually been fetched (any row has actual_points)
    fetched = any(p.get("actual_points") is not None for p in players)

    # Summary stats
    scored = [p for p in players if p.get("actual_points") is not None]
    if scored:
        errors = [abs((p["actual_points"] or 0) - (p["predicted_points"] or 0)) for p in scored]
        correct_count = sum(1 for p in scored if p.get("correct") == 1)
        summary = {
            "count": len(scored),
            "correct": correct_count,
            "directional_accuracy": round(correct_count / len(scored), 4) if scored else None,
            "mae": round(sum(errors) / len(errors), 2) if errors else None,
        }
    else:
        summary = {"count": len(players), "correct": 0, "directional_accuracy": None, "mae": None}

    # Add model_predicted_over flag for badge rendering in UI
    for p in players:
        p["model_predicted_over"] = 1 if (p.get("predicted_over_prob") or 0) > 0.5 else 0

    return {
        "game_date": target,
        "fetched": fetched,
        "summary": summary,
        "players": players,
    }


@app.post("/api/results/fetch")
async def fetch_daily_results(game_date: str | None = None):
    """
    Pull actual NBA results for game_date from the NBA API and score predictions.
    Does NOT retrain models — use /api/scoring/run for the full learning pipeline.
    """
    target = game_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    result = score_past_predictions(DB_PATH, target)
    return result


# ── Injury endpoints ─────────────────────────────────────────────────────────

@app.get("/api/injuries")
async def get_injuries(game_date: str | None = None):
    """Return all injury records for a given date (defaults to today)."""
    target = game_date or date.today().strftime("%Y-%m-%d")
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT player_id, player_name, team_abbreviation, status, description
            FROM injury_status
            WHERE fetched_date = ?
            ORDER BY team_abbreviation ASC, status ASC
            """,
            (target,),
        ).fetchall()
    finally:
        conn.close()
    return {
        "date": target,
        "count": len(rows),
        "injuries": [_row_to_dict(r) for r in rows],
    }


@app.get("/api/injuries/team/{team_abbr}")
async def get_team_injuries(team_abbr: str, game_date: str | None = None):
    """Return injury report for a specific team."""
    target = game_date or date.today().strftime("%Y-%m-%d")
    if get_team_injury_report is None:
        raise HTTPException(status_code=503, detail="Injury module not available")
    report = get_team_injury_report(team_abbr, DB_PATH, target)
    return {"team": team_abbr.upper(), "date": target, "players": report}


@app.post("/api/injuries/fetch")
async def trigger_injury_fetch(background_tasks: BackgroundTasks, game_date: str | None = None):
    """Trigger injury data fetch from ESPN for all 30 teams."""
    if fetch_and_store_injuries is None:
        raise HTTPException(status_code=503, detail="Injury module not available")
    target = game_date or date.today().strftime("%Y-%m-%d")

    def _do_fetch():
        count = fetch_and_store_injuries(DB_PATH, target)
        logger.info(f"Injury fetch complete: {count} records for {target}")

    background_tasks.add_task(_do_fetch)
    return {"status": "started", "date": target}


# ── RL agent status endpoint ──────────────────────────────────────────────────

@app.get("/api/trader/rl-status")
async def get_rl_status():
    """Return RL agent stats: Q-table size, epsilon, greedy action distribution."""
    if RLBettingAgent is None:
        raise HTTPException(status_code=503, detail="RL agent module not available")
    agent = RLBettingAgent.load()
    return agent.get_stats()


# ── Health / stats endpoint ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Basic health check with DB stats."""
    conn = _get_conn()
    stats = {}
    for table in ["player_game_logs", "predictions", "results", "sportsbook_lines"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except Exception:
            stats[table] = 0
    conn.close()
    return {"status": "ok", "db_stats": stats, "timestamp": datetime.now().isoformat()}


@app.get("/api/players/active")
async def get_tracked_players():
    """Return players currently tracked in the database."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT player_id, player_name, COUNT(*) AS game_count,
               MAX(game_date) AS last_game
        FROM player_game_logs
        GROUP BY player_id, player_name
        ORDER BY game_count DESC
        """
    ).fetchall()
    conn.close()
    return {"players": [_row_to_dict(r) for r in rows]}
