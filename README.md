# NBA Points Prediction Model

Predicts NBA player points over/under using an ensemble of ML models (Linear Regression, Random Forest, XGBoost) and compares the predictions against live sportsbook and prediction-market lines to surface positive expected-value opportunities.

**Live stats dashboard:** https://vedaantk.github.io/nba-predictor/
**Prompt log:** [prompt_log.md](prompt_log.md) · **Full setup guide:** [SETUP.md](SETUP.md)

---

## How the API is called

Sportsbook lines come from [The Odds API](https://the-odds-api.com), called over plain HTTP with Python's `requests` module in [`ingestion/odds_api.py`](ingestion/odds_api.py) — no vendor SDK. Two endpoints are used: `GET /v4/sports/basketball_nba/events` to list upcoming games, then `GET /v4/sports/basketball_nba/events/{event_id}/odds` for each game, with the query parameters `apiKey`, `regions=us`, `markets=player_points`, and `oddsFormat=american`. Both return JSON: the events endpoint gives a list of objects with `id`, `commence_time` (an ISO 8601 UTC string), and team names, while the odds endpoint returns a nested structure of `bookmakers → markets → outcomes`, where each outcome holds a player name (string), a `point` line (float), and an American `price` (int) — with Over and Under arriving as two separate entries that the parser recombines into one record per player. Those records are converted to implied probabilities via `american_odds_to_prob()` and written to a SQLite table. Three other sources feed the same pipeline: the NBA Stats API for player game logs (through the `nba_api` wrapper package), plus Kalshi and Polymarket for prediction-market prices — all three are keyless public endpoints.

Every response is checked for HTTP errors with `raise_for_status()` inside a `try/except requests.RequestException`, under a `(5, 8)` second connect/read timeout, so a network failure, a 401 from a bad key, or a 429 rate-limit logs a message and returns an empty list rather than raising.

---

## API key setup

Only The Odds API requires a key. The NBA Stats, Kalshi, and Polymarket endpoints are keyless.

1. Get a free key at **[the-odds-api.com](https://the-odds-api.com)** — the free tier allows **500 requests/month**, which is enough for daily use.
2. Copy the template and paste your key in:
   ```bash
   cp .env.example .env     # Windows: copy .env.example .env
   ```
   ```
   ODDS_API_KEY=your_key_here
   ```
3. `config/settings.py` loads it from the environment via `python-dotenv`.

> `.env` is listed in `.gitignore` and is never committed. Only `.env.example`, which holds a placeholder, is tracked. Never put the key in the dashboard's JavaScript — anyone can read it from browser devtools. The GitHub Pages dashboard avoids this entirely by reading a static `data.json` snapshot exported from the local database, so no keyed endpoint is ever called from the browser.

### Staying inside the free tier

Each event's props cost 1 credit, so the client in `ingestion/odds_api.py` enforces a budget: it reads the `x-requests-remaining` header the API returns on every response and persists it to `cache/odds_api_budget.json`, stops fetching once the balance reaches a reserve floor, caps the number of credit-spending runs per calendar day, and caches responses to disk — permanently for past dates, and on a TTL for today.

---

## Quickstart

```bash
git clone https://github.com/VedaantK/nbaPred.git
cd nbaPred

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your Odds API key into .env

python -m scripts.backfill_data   # one-time, ~20 min (NBA API rate limits)
python train.py                   # trains the ensemble, ~2-10 min
uvicorn api.server:app --reload --port 8000
```

Then in a second terminal:

```bash
cd dashboard && npm install && npm run dev
```

Open http://localhost:5173.

For a step-by-step walkthrough including Windows commands and troubleshooting, see **[SETUP.md](SETUP.md)**.

---

## Project Structure

```
nbaPred/
├── config/          # Settings, DB init, logging
├── ingestion/       # NBA API, Odds API, Kalshi, Polymarket, injuries
├── features/        # Feature engineering (35+ features)
├── models/          # Training, prediction, evaluation, RL paper trader
├── api/             # FastAPI backend (port 8000)
├── dashboard/       # React + Vite frontend (port 5173)
├── scripts/         # Backfill, daily pipeline, scoring, dashboard export
└── utils/           # Name matching across data sources
```

---

## Daily Workflow

Run this each day before games start (~11 AM Eastern):

```bash
python -m scripts.daily_pipeline
```

This updates game logs with recent results, fetches sportsbook lines from The Odds API, fetches Kalshi and Polymarket markets, and generates predictions with edge calculations.

Run this each morning after games finish (~8 AM):

```bash
python -m scripts.score_results
```

This scores predictions against actual results, logs performance, and retrains the models. Pass `--date YYYY-MM-DD` to score a specific day (defaults to yesterday).

To refresh the public dashboard snapshot:

```bash
python -m scripts.export_dashboard
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/predictions/today` | Today's predictions sorted by edge |
| `GET /api/predictions/history?days=30` | Historical predictions with outcomes |
| `GET /api/performance/summary?days=30` | Accuracy, ROI, calibration metrics |
| `GET /api/performance/calibration` | Calibration curve data |
| `GET /api/performance/comparison?days=30` | Model vs sportsbook vs markets |
| `GET /api/features/importance?model=xgboost` | Feature importance rankings |
| `POST /api/pipeline/run` | Trigger daily pipeline manually |
| `GET /api/health` | Health check + DB row counts |

---

## Features Used (35+)

**Scoring**: rolling averages (3/5/10/20 games), season avg, median, std, trend slope
**Shooting**: FGA, FG%, 3PA, 3P%, FTA, FT%
**Usage**: minutes trend, usage rate, back-to-back indicator, rest days
**Opponent**: defensive rating, pace, points allowed
**Derived**: hot streak ratio, consistency score, matchup-adjusted average

---

## Edge Calculation

```
edge = model_over_probability - sportsbook_implied_probability
```

Confidence levels:
- **High**: all 3 models agree + edge > 10%
- **Medium**: 2/3 models agree + edge > 5%
- **Low**: models disagree or edge < 5%

Over-probability is estimated by adding historical model residuals to the point prediction and computing what fraction of simulated outcomes exceed the line.
