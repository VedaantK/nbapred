"""
Central configuration for the entire project.
All API keys, database paths, and constants live here.
"""

import os

# xgboost and torch each bundle their own OpenMP runtime. Two of them live in
# one process only if OMP stays single-threaded — otherwise the first torch
# call after an xgboost fit deadlocks (0% CPU, no traceback). This must run
# before anything imports either library; config.settings is imported first by
# every entrypoint, so this is the earliest safe place.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nba_predictor.db"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# API Keys (set in .env file — see .env.example)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Kalshi API (public, no key needed for market data)
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

# Polymarket CLOB API (public, no key needed for market data)
POLYMARKET_BASE_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# NBA API settings
NBA_SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]  # 6 seasons for training data
REQUEST_DELAY = 0.6  # Seconds between nba_api requests (rate limit)

# ── The Odds API budget ───────────────────────────────────────────────────────
# Cost model (verified against the v4 docs):
#   /sports/{sport}/events            -> free, does not count against quota
#   /events/{id}/odds                 -> [markets] x [regions] = 1 credit per game
# One market (player_points) in one region (us) means a full daily fetch costs
# exactly one credit per game on the slate. At ~7.2 games/day across a 170-day
# season that is ~216 credits/month, or 43% of the 500 free tier.
ODDS_CACHE_TTL_HOURS = 6        # today's lines; was 2, which allowed 12 refreshes/day
ODDS_API_RESERVE_CREDITS = 50   # stop fetching once the balance falls this low
ODDS_API_MAX_EVENTS_PER_RUN = 20  # a normal slate is 2-13; more than this means something is wrong
ODDS_API_MAX_RUNS_PER_DAY = 3   # guards the dashboard's "Run pipeline" button

# Model settings
ROLLING_WINDOWS = [3, 5, 10, 20]  # Game windows for rolling averages
MIN_GAMES_FOR_PREDICTION = 10  # Player must have this many games before we predict
EDGE_THRESHOLD = 0.08  # 8% probability edge to flag a bet
CONFIDENCE_THRESHOLD = 0.60  # 60% confidence minimum to surface prediction

# Top players to track (starters + key bench scorers)
TOP_PLAYERS_COUNT = 150
