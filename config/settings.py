"""
Central configuration for the entire project.
All API keys, database paths, and constants live here.
"""

import os
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

# Model settings
ROLLING_WINDOWS = [3, 5, 10, 20]  # Game windows for rolling averages
MIN_GAMES_FOR_PREDICTION = 10  # Player must have this many games before we predict
EDGE_THRESHOLD = 0.08  # 8% probability edge to flag a bet
CONFIDENCE_THRESHOLD = 0.60  # 60% confidence minimum to surface prediction

# Top players to track (starters + key bench scorers)
TOP_PLAYERS_COUNT = 150
