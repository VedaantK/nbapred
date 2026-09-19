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

> Can you look at this github repo https://github.com/VedaantK/nbaPred.git and make sure that everything is just running ok, and there are no major bugs errors. igure out how to integrate the Odds API so that you can compare your predictions with the SportsBook.Then tell me two things. Also I want to add this to this website of mine so do that as well. https://github.com/VedaantK/VedaantK.github.io.git

This one mattered most. Asking for an audit rather than a feature surfaced four real bugs that I would not have found by reading the code myself: dead query parameters being sent to The Odds API events endpoint, look-ahead bias in the feature engineering (rolling averages were including the game being predicted), a crash on date handling, and no guard at all on the 500-request monthly quota. These became commit `856a248`.

### 2. Get a running system back from a broken checkout

> Can you outline everything I need to do, to get the system up and running again, and all the system fully running.

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

### 7. Shorten the README to meet the assignment's grading rubric

> Can you edit the read me to be shorter but still meets all of this critera README: In the repo README, include 3–5 sentences that explain (at a high level) how the API is called (for example, what modules are used if any, what key parameters are provided, and what format / data types are returned). If your code requires an API key or other authentication, provide information on how to obtain and use that API key (WITHOUT exposing the key itself in the repo). Also include brief instructions for running your code: what to install, and what command or file to run.

First pass just trimmed prose inside the existing six-heading structure. A follow-up —

> No, It should only be 3-5 sentances though, that explain the whole project at a high level and also the use of the API

— clarified that the 3-5 sentence limit was meant for the entire opening explanation, not a separate constraint layered on top of the existing sections. Lesson: "shorter" is ambiguous about scope, and a rubric quoted back verbatim doesn't mean each clause deserves its own heading.

### 8. Confirm the API key never touched git history

> it says this whole paragraph about how the API key should never be posted to github and I know right now it is in the gitignore but how do I make sure it has never been pushed

(Prompt continued with the assignment's pasted privacy-note paragraph about zero-tolerance for committed secrets.) `.gitignore` only prevents *future* accidental commits, so instead of just re-reading the ignore rule I searched the actual history: `git log --all --full-history` for a `.env` file ever being added, and `git log --all -p` grepped for the literal key value read from the local `.env` (without ever printing the key itself in output). Came back clean on both.

### 9. Add more data to the dashboard

> add some more data to the github page, maybe from last season any predictuons or anything else that can be added

Rather than invent new numbers, I checked what `export_dashboard.py` was already collecting and found a `per_season` row-count breakdown already sitting in `data.json`, unrendered. Added a "Games Per Season" chart to surface that real, already-computed data instead of fabricating something new. (Removed again in item 11, once the dashboard was being finalized and that section no longer earned its place.)

### 10. Ask whether last season's data could demonstrate the model working

> Is there any data that you have from last season maybe that could show the model working or not since it is currently offsseason

Best prompt of the session — phrased as a question, which forced an actual investigation instead of a quick edit. Tracing `models/train.py`'s time-based 80/20 split against the per-season row counts showed the held-out test window lands almost entirely inside the 2025-26 season (2025-11-07 to 2026-04-12). The MODEL ACCURACY numbers already on the dashboard were a genuine last-season backtest — just mislabeled as a generic "test set." Persisted `test_start`/`test_end` into `metrics.json` (rather than hardcoding the one date range already computed) so future retrains keep the label accurate automatically.

### 11. Finalize the dashboard for submission

> Yes, commit and push it all. Also get the github page fully finihsed ready for the final submisson. Get rid of the Games per season section.

"Fully finished" prompted a full re-read of the page rather than just the one requested removal, which caught a real inconsistency: the intro paragraph named only 3 of the 4 trained models, missing the neural net that the MODEL ACCURACY section was already rendering.

### 12. Update this log, and drop unimplemented sources from the dashboard

> Can you update the prompt log to add all the new things we talked about. Also since there is no Kalshi or Polymarket integration right now you can get rid of it from the github pages.

Good instinct on the second half — the DATA SOURCES panel was showing Kalshi and Polymarket permanently red ("awaiting fix", 0 rows), which reads as broken rather than simply not built yet. Removed both from `collect_sources()` in `export_dashboard.py` and re-exported `data.json`, rather than special-casing them out in the page's JS, so the static site and any future local export stay driven by one source of truth.

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
