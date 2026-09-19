# Prompt Log

AI tools used to build and debug this project.

## Tools / Models

| Tool | Model | Used for |
|------|-------|----------|
| Claude Code (CLI) | Claude Opus 5 | Debugging the data pipeline, API quota management, the static stats dashboard, portfolio integration |
| Claude (chat) | — | Original build of the ML ensemble, ingestion layer, and FastAPI backend |

**Note on coverage:** the prompts below are from the September 2026 debugging and deployment session, which I have a verbatim record of. The original build (commits `7b18041` through `a9fbabd`, March–April 2026) was done on a different machine and I no longer have that transcript, so those prompts are summarized from memory in the last section rather than quoted.

---

## Session: debugging, quota control, and deployment (Sept 2026)

### 1. Audit the existing repo and identify the APIs

> Can you look at this github repo https://github.com/VedaantK/nbaPred.git and make sure that everything is just running ok, and there are no major bugs or API errors. Then tell me two things. How hard would it be to have this project along with the NBA also apply to the NFL and also tell me what API's this project uses. Also I want to add this to this website of mine so do that as well. https://github.com/VedaantK/VedaantK.github.io.git

This one mattered most. Asking for an audit rather than a feature surfaced four real bugs that I would not have found by reading the code myself: dead query parameters being sent to The Odds API events endpoint, look-ahead bias in the feature engineering (rolling averages were including the game being predicted), a crash on date handling, and no guard at all on the 500-request monthly quota. These became commit `856a248`.

### 2. Get a running system back from a broken checkout

> Can you outline everything I need to do, to get the system up and running again, and all the system fully running

Asking for an ordered checklist instead of "fix it" gave me something I could actually follow across two machines, and it became the basis for `SETUP.md`.

### 3. Quota strategy for a free-tier API

> I installed OpenMP, I don't know how to kill the training deadlock can you tell me where this project is in my finder, I have my Odds API key can you tell me where to paste it and also figure out a plan to not max out its 500 credits a month. Fix everything in phase B yourself. Then we can work on the next two phases.

The most useful prompt for the API integration specifically. Naming the concrete constraint — 500 requests/month, free tier — produced the whole budget system in `ingestion/odds_api.py` rather than a generic "add caching" answer:

- Read the `x-requests-remaining` header the API returns on *every* response and persist it to `cache/odds_api_budget.json`, instead of logging it and throwing it away.
- Stop fetching entirely once the balance hits a reserve floor (`ODDS_API_RESERVE_CREDITS`).
- Cap runs per calendar day, because the dashboard exposes the pipeline as a button with no rate limit — at the original 2-hour cache that was ~90 credits/day from clicking alone.
- Cache historical dates forever and only apply a TTL to today, since past odds never change.

### 4. Approve and extend the quota fixes

> yes implement those three credit changes, i also added the API. Now if you have everything needed make a dashboard for the predictor and add it to the github pages

### 5. Static dashboard for the portfolio

> Can you make a pretty quick and simple dashboard for this, where I can see some stats relating to the project

This is where the architecture decision got made. Because The Odds API needs a key, the key can never reach browser JavaScript — anyone can open devtools and read it. The answer was `scripts/export_dashboard.py`, which runs locally against the database and writes a static `data.json` that GitHub Pages serves. The live page at `/nba-predictor/` reads only that snapshot and never touches a keyed endpoint.

### 6. Ship it

> Can you first commit and push it to the website, we can fix the problems later

---

## Original build (March–April 2026, summarized from memory)

No verbatim transcript. The prompts that shaped the initial implementation were roughly:

- Asking how The Odds API's player props endpoints are structured, and having each field of a sample JSON response explained before writing any parsing code — the `bookmakers → markets → outcomes` nesting is not obvious, and `outcomes` splits Over and Under into two separate entries that have to be recombined per player.
- Asking how to convert American odds into implied probability, which became `american_odds_to_prob()`.
- Asking for a feature set for predicting player points from game logs, which produced the rolling 3/5/10/20-game windows, rest days, back-to-back flags, and opponent defensive rating.
- Asking how to match player names across four sources (NBA Stats, The Odds API, Kalshi, Polymarket) that all spell them differently, which became `utils/`.
- Asking how to structure a time-based train/test split so the model is never evaluated on games earlier than the ones it trained on.

## What I verified rather than trusted

AI-generated API code was checked against live responses before it was kept:

- Printed raw JSON from each endpoint and confirmed the field names actually matched what the parsing code expected.
- Confirmed the events endpoint ignores `commenceTimeFrom`/`commenceTimeTo`, which is why `get_todays_nba_events()` fetches all upcoming events and filters client-side by `commence_time` instead.
- Tested failure paths directly: wifi off (caught by `requests.RequestException` with a `(5, 8)` timeout), an empty slate with no games, and events returned without IDs. All log and return empty rather than raising.
