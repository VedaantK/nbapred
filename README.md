# NBA Points Prediction Model

Predicts NBA player points over/under using an ensemble of ML models (Linear Regression, Random Forest, XGBoost) and compares predictions against live sportsbook and prediction-market lines to surface positive expected-value opportunities. Sportsbook lines are pulled from [The Odds API](https://the-odds-api.com) using Python's `requests` module in [`ingestion/odds_api.py`](ingestion/odds_api.py), passing an `apiKey`, `regions`, `markets=player_points`, and `oddsFormat` as query parameters. The API returns JSON containing nested `bookmakers → markets → outcomes` objects, each with a player name (string), a `point` line (float), and an American `price` (int), which the parser converts into implied probabilities and stores in SQLite. Player game logs are pulled the same way from the keyless `nba_api` package, and Kalshi/Polymarket supply keyless prediction-market prices for comparison.

**Live dashboard:** https://vedaantk.github.io/nba-predictor/ · **Full setup guide:** [SETUP.md](SETUP.md)

---

## API key setup

Only The Odds API requires a key (NBA Stats, Kalshi, and Polymarket are keyless).

1. Get a free key at **[the-odds-api.com](https://the-odds-api.com)** (500 requests/month, free tier).
2. `cp .env.example .env` and paste your key in as `ODDS_API_KEY=your_key_here`.
3. `config/settings.py` loads it from the environment via `python-dotenv`.

`.env` is gitignored and never committed — only the placeholder `.env.example` is tracked.

---

## Quickstart

```bash
git clone https://github.com/VedaantK/nbapred.git
cd nbapred

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your Odds API key into .env

python -m scripts.backfill_data   # one-time, ~20 min
python train.py                   # trains the ensemble, ~2-10 min
uvicorn api.server:app --reload --port 8000
```

In a second terminal:

```bash
cd dashboard && npm install && npm run dev
```

Open http://localhost:5173. For Windows commands and troubleshooting, see **[SETUP.md](SETUP.md)**.
