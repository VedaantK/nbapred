# NBA Points Prediction Model

Predicts NBA player points over/under using ML models (Linear Regression, Random Forest, XGBoost) and compares against sportsbook + prediction market lines to find positive expected value opportunities.

## Project Structure

```
nba-predictor/
├── config/          # Settings, DB init, logging
├── ingestion/       # NBA API, Odds API, Kalshi, Polymarket
├── features/        # Feature engineering (35+ features)
├── models/          # Training, prediction, evaluation
├── api/             # FastAPI backend (port 8000)
├── dashboard/       # React + Vite frontend (port 5173)
├── scripts/         # Backfill, daily pipeline, scoring
└── utils/           # Name matching across data sources
```

## Setup

### 1. Python environment

```bash
cd nba-predictor
pip install -r requirements.txt
```

### 2. Node.js (for the dashboard)

```bash
cd dashboard
npm install
```

---

## Running the System

### Step 1 — Backfill historical data (run once, ~20 min)

```bash
cd nba-predictor
python -m scripts.backfill_data
```

This pulls 2 seasons of game logs for ~150 players from the NBA API.

### Step 2 — Train the models

```bash
python train.py
```

Trains Linear Regression, Random Forest, and XGBoost with time-based train/test split. Saves models to `models/saved/`.

### Step 3 — Start the API server

```bash
uvicorn api.server:app --reload --port 8000
```

### Step 4 — Start the dashboard

```bash
cd dashboard
npm run dev
```

Open http://localhost:5173

---

## Daily Workflow

Run this each day before games start (~11 AM Eastern):

```bash
python -m scripts.daily_pipeline
```

This:
1. Updates game logs with recent results
2. Fetches sportsbook lines (The Odds API)
3. Fetches Kalshi + Polymarket markets
4. Generates predictions with edge calculations

Run this each morning after games finish (~8 AM):

```bash
python -m scripts.score_results
```

This scores predictions against actual results and logs performance. Add `--retrain` to force a model retrain.

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
