# NBA Predictor — Setup Guide

Full-stack NBA player points predictor with ML ensemble, prediction market integration, injury tracking, and RL paper trader.

- **Backend**: FastAPI + SQLite (`localhost:8000`)
- **Frontend**: Vite + React + Tailwind CSS (`localhost:5173`)

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.11+ | python.org |
| Node.js | 18+ | nodejs.org |
| Git | any | git-scm.com |

---

## 1. Clone the Repository

```bash
git clone https://github.com/VedaantK/nbaPred.git
cd nbaPred
```

---

## 2. Python Environment

### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **PyTorch note**: `requirements.txt` installs the CPU version of PyTorch automatically. This is sufficient for training and inference. If you have an NVIDIA GPU and want faster training, install the GPU build from pytorch.org after the step above.

---

## 3. Environment Variables

Copy the example file and fill in your API key:

### Mac / Linux
```bash
cp .env.example .env
```

### Windows
```cmd
copy .env.example .env
```

Then open `.env` and set your key:
```
ODDS_API_KEY=your_key_here
```

Get a free API key at **the-odds-api.com** (free tier: 500 requests/month, enough for daily use).

> Kalshi and Polymarket data is fetched from their public APIs — no key needed.

---

## 4. Initialize the Database

```bash
python -c "from config.db import init_db; init_db()"
```

This creates `data/nba_predictor.db` with all required tables.

---

## 5. Backfill Historical Data

Pull 3 seasons of player game logs and team stats (needed to train models):

```bash
python -m scripts.backfill_data
```

This makes many requests to the NBA API with rate limiting — expect **10–30 minutes**. Run it once; it uses `INSERT OR IGNORE` so re-running is safe.

---

## 6. Train the Models

```bash
python train.py
```

Trains all 4 models (Linear Regression, Random Forest, XGBoost, Neural Net) and saves them to `models/saved/`. Takes **2–10 minutes** on CPU.

---

## 7. Start the Backend

### Mac / Linux
```bash
uvicorn api.server:app --reload --port 8000
```

### Windows
```cmd
uvicorn api.server:app --reload --port 8000
```

The API is now running at `http://localhost:8000`. Keep this terminal open.

---

## 8. Start the Dashboard

Open a **second terminal** window, activate the same virtual environment, then:

### Mac / Linux
```bash
cd dashboard
npm install
npm run dev
```

### Windows
```cmd
cd dashboard
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 9. Daily Workflow

Once set up, each day:

1. **Fetch Odds** — Sidebar → "Fetch Today's Odds" (pulls sportsbook lines + Kalshi/Polymarket markets)
2. **Run Predictions** — Sidebar → "Run Predictions" (generates over/under predictions for star players)
3. **View Predictions** — Dashboard home shows today's predictions with edge and confidence
4. **End of Day** — Go to "Daily Results" → "Pull Results" to fetch actual NBA scores
5. **Daily Learning** — Click "Run Daily Learning" to update ensemble weights (~15 seconds)
6. **Weekly** — Click "Full Retrain" once a week to rebuild all models from scratch (5–30 min)

---

## Project Structure

```
nbaPred/
├── api/                  FastAPI server (server.py)
├── config/               Settings, DB schema, logging
├── dashboard/            Vite + React + Tailwind frontend
│   └── src/pages/        Dashboard pages (Predictions, Daily Results, Paper Trader, etc.)
├── features/             Feature engineering (rolling stats, injury features)
├── ingestion/            Data fetchers (NBA API, odds, Kalshi, Polymarket, injuries)
├── models/               ML models (train, predict, evaluate, neural net, RL agent, paper trader)
├── scripts/              Utility scripts (backfill, daily pipeline, score results)
├── utils/                Name matching, helpers
├── .env.example          Environment variable template
├── requirements.txt      Python dependencies
└── SETUP.md              This file
```

---

## Troubleshooting

**`ODDS_API_KEY` not found / empty**
Make sure `.env` exists in the project root and contains `ODDS_API_KEY=your_key`.

**`No trained models found`**
Run `python train.py` first. You also need historical data — run `python -m scripts.backfill_data` before training.

**NBA API timeouts**
The NBA API rate-limits aggressively. If `backfill_data.py` fails partway through, just re-run it — already-stored rows are skipped.

**`torch` import error**
If PyTorch fails to install on your machine (rare), the system gracefully falls back to 3 models (no neural net). Everything else works normally.

**Port already in use**
If port 8000 is taken: `uvicorn api.server:app --reload --port 8001`
Then update `dashboard/vite.config.js` proxy target to `http://localhost:8001`.

**Windows: `.venv\Scripts\Activate.ps1` blocked by execution policy**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Then re-run the activate command.
