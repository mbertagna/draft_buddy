# Draft Buddy

Draft Buddy uses Docker Compose as the primary local workflow for the refactored runtime boundaries:

- `web`: FastAPI app and session management
- `rl`: Gym environment, feature extraction, rewards, models, and training
- `data`: player loading and projection generation
- `simulator`: stateless season evaluation
- `core`: shared draft state, controller, rules, bots, and entities
- `arch_viz`: architecture visualization tooling

## Docker Compose Usage

### Prerequisites

- Docker
- Docker Compose

### First-Time Setup

Copy environment variables and set your active league:

```bash
cp .env.example .env
```

League profiles live under `config/leagues/` with per-season overlays in `config/seasons/`. Set these in `.env`:

```bash
DRAFT_BUDDY_LEAGUE=red_league_10      # ESPN Red League (10-team, full PPR)
DRAFT_BUDDY_SEASON=2026
```

To switch to Redraft NBFL (12-team Sleeper, half PPR) later:

```bash
DRAFT_BUDDY_LEAGUE=redraft_nbfl_12
```

Each league uses its own generated player CSV under `data/leagues/{league_id}/generated/{year}/`. Run `docker compose run --rm data` after switching leagues so projections match that league's scoring.

Build the image used by every service:

```bash
docker compose build
```

Each service bind-mounts the repository into `/app` and runs with `PYTHONPATH=/app/src`, so generated files are written back to your host checkout.

Common output locations on the host:

- `data/`: generated player data and draft state files
- `logs/`: training metrics, dashboards, and run logs
- `models/`: checkpoints and trained model artifacts
- `coverage.xml`: XML coverage report from `test-cov`
- `htmlcov/`: HTML coverage report from `test-cov`
- `viz/`: Mermaid architecture output from `ast`

### Services

| Service | Purpose | Default command |
| --- | --- | --- |
| `webapp` | Run the FastAPI web application | `python scripts/run_webapp.py` |
| `train` | Run RL training | `python scripts/train.py` |
| `test` | Run the test suite | `python -m pytest tests/` |
| `test-cov` | Run tests with coverage outputs | `python -m pytest tests/ --cov=src/draft_buddy ...` |
| `data` | Generate player projections and merged draft data | `python scripts/generate_projections.py --year 2026` |
| `ast` | Generate Mermaid architecture diagrams | `python -m draft_buddy.arch_viz.cli --project-root /app --output-dir /app/viz --all-default-entries --strategy module` |
| `insights-search` | Fetch web search snippets for top 150 ADP players (Valyu default) | `python scripts/fetch_player_insight_search.py --year 2026 --top-n 150 --search-provider valyu` |
| `insights-synthesize` | Synthesize Gemini Flash player insights from cached search | `python scripts/synthesize_player_insights.py --year 2026 --top-n 150` |
| `position-guide` | Generate static RL position probability cheat sheet | `python scripts/generate_position_guide.py --simulations 5000` |

### Common Commands

Start the web application:

```bash
docker compose up webapp
```

In the header, use **Sim → Bot | Policy** to choose the engine for **Sim Pick** and **Auto Draft**. Bot uses configured heuristic/ADP strategies; Policy uses the loaded RL checkpoint (`MODEL_PATH_TO_LOAD`).

Run training:

```bash
docker compose run --rm train
```

Generate training plots from the latest CSV metrics without training:

```bash
docker compose run --rm train python scripts/train.py -p
```

Run the test suite:

```bash
docker compose run --rm test
```

Run tests with coverage:

```bash
docker compose run --rm test-cov
```

Generate player projections with the default compose command:

```bash
docker compose run --rm data
```

Veteran weekly stats are downloaded from nflverse's current `stats_player` release as per-season files (`stats_player_week_{year}.csv`) into `data/cache/nflverse/`. The lookback window defaults to two completed seasons (`Config.data.LEGACY_STATS_LOOKBACK_SEASONS`); override with `--lookback-seasons`.

Override the data-generation command:

```bash
docker compose run --rm data python scripts/generate_projections.py --year 2024 --rookie_projection_method hybrid
```

Generate architecture diagrams:

```bash
docker compose run --rm ast
```

### Player Insights (manual pre-draft enrichment)

Offline player insight enrichment is a **manual, two-step** pipeline that prepares research-backed outlook data for the draft UI (see [PLAYER_INSIGHTS_PART2_PLAN.md](PLAYER_INSIGHTS_PART2_PLAN.md)).

**Prerequisites:**

1. Copy `.env.example` to `.env` and set:
   - `VALYU_API_KEY` from [Valyu](https://platform.valyu.ai/) (default search provider)
   - `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/) and/or `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai/)
2. Optional: for Google CSE instead, set `INSIGHTS_SEARCH_PROVIDER=google`, `GOOGLE_CSE_API_KEY`, and `GOOGLE_CSE_ID` (note: CSE is closed to new customers and sunsets Jan 2027).

**Run order:**

```bash
# 1. Generate player projections (if not already done)
docker compose run --rm data

# 2. Fetch and cache search snippets (Valyu by default)
docker compose run --rm insights-search

# 3. Synthesize structured insights (Gemini or OpenRouter)
docker compose run --rm insights-synthesize
```

**Synthesis model selection:**

Defaults come from `.env` (`INSIGHTS_LLM_PROVIDER`, `INSIGHTS_LLM_MODEL`). Override per run:

```bash
docker compose run --rm insights-synthesize python scripts/synthesize_player_insights.py \
  --provider openrouter --model deepseek/deepseek-v4-flash
```

**Outputs:**

- Search cache: `data/cache/insights/search/{sleeper_id}/`
- Synthesis cache: `data/cache/insights/synthesis/{sleeper_id}.json`
- Merged insights export: `data/insights/exports/player_insights_{year}_{timestamp}.json`

Each synthesis run writes a new timestamped export file. The webapp loads the newest export by filename timestamp at startup. Legacy undated `data/player_insights_{year}.json` files are used as a fallback when no exports exist yet.

**Partial re-runs:**

```bash
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --max-players 20 --start-index 0
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --force
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --search-provider google --force
docker compose run --rm insights-synthesize python scripts/synthesize_player_insights.py --force
```

### Live Draft Assistant

The web UI includes an on-demand **LLM draft assistant** alongside the fast RL position chips. Set `GEMINI_API_KEY` and/or `OPENROUTER_API_KEY` in `.env`.

**Supported models:** Gemini 2.5 Flash, Gemini 2.5 Flash Lite, DeepSeek V4 Pro, DeepSeek V4 Flash (via OpenRouter).

**In the header:**

- **Auto assistant** — when on, fires once per snake turn when scope allows (skipped during clock overrides)
- **Scope** — *My picks only* (agent team from league config) or *Every team* (auto only)
- **My model / Others** — separate model pickers for your team vs other teams (defaults from `ADVISOR_AGENT_MODEL` / `ADVISOR_OTHER_TEAMS_MODEL`)
- **Ask Assistant** — always available during an active draft; uses the selected/on-clock team (including overrides)

The assistant builds per-position shortlists (top 7 by VORP/ADP for the RL model's top two positions, top 5 for the others) and returns a structured pick recommendation. Min GP Frac from the player table is sent with each request.

See [PLAYER_INSIGHTS_PART2_PLAN.md](PLAYER_INSIGHTS_PART2_PLAN.md) for architecture details.

### Position Guide (pre-draft cheat sheet)

Offline Monte Carlo simulation produces a **static position probability guide** for your draft slot — useful as a fallback when the live dashboard is unavailable. Defaults (`num-teams`, `slot`, `checkpoint-dir`) come from the active league profile in `.env`.

**Prerequisites:**

1. Generate player projections for the active league: `docker compose run --rm data`
2. A trained policy checkpoint for that league size (paths are set in `config/seasons/{league}_{year}.json`)

**Run (uses active league from `.env`):**

```bash
docker compose run --rm position-guide
open data/guides/exports/position_guide_*teams_slot*_*_*.html
```

**Override league temporarily:**

```bash
DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 docker compose run --rm position-guide
```

**Outputs:**

- JSON: `data/guides/exports/position_guide_{num_teams}teams_slot{slot}_{year}_{timestamp}.json`
- HTML: same basename with `.html` (printable cheat sheet)

Each run writes a new timestamped export. Filenames include league size so 10-team and 12-team guides do not collide.

### League profiles and season rollover

| League | Profile ID | Platform | Scoring | 2026 draft slot |
| --- | --- | --- | --- | --- |
| Red League | `red_league_10` | ESPN | Full PPR | 2 |
| Redraft NBFL | `redraft_nbfl_12` | Sleeper | Half PPR | 5 |

**Season rollover checklist** (each August):

1. Copy `config/seasons/{league}_2026.json` to `{league}_2027.json`
2. Update `season`, `bye_weeks`, `draft.AGENT_START_POSITION`, and checkpoint paths
3. Run `docker compose run --rm data` for each league you use
4. Train or point `training.MODEL_PATH_TO_LOAD` at the correct `models/{N}_teams_*` checkpoint

Player projections only include nflverse-trackable scoring rules. Bonuses without reliable stat columns (50+ yard TDs, D/ST details, IR slots) are omitted per league JSON.

### `up` vs `run --rm`

Use `docker compose up` for long-running services that should stay attached to a port, such as `webapp`.

Use `docker compose run --rm` for one-off tasks such as training, tests, coverage, data generation, and architecture visualization. The `--rm` flag removes the container when the command exits.

Because every service uses `working_dir: /app`, command overrides run from the repository root inside the container. That means overrides like:

```bash
docker compose run --rm test python -m pytest tests/test_config.py
```

behave consistently across services.

### Accessing Outputs

- Web UI: [http://localhost:5001](http://localhost:5001)
- Coverage HTML report: [htmlcov/index.html](htmlcov/index.html)
- Architecture diagrams: `viz/`
- Training logs and dashboards: `logs/`
- Model checkpoints: `models/`

## Package Structure

```text
.
├── data/
├── frontend/
├── logs/
├── models/
├── scripts/
├── src/draft_buddy/
│   ├── arch_viz/
│   ├── core/
│   ├── data/
│   ├── rl/
│   ├── simulator/
│   └── web/
├── config/
│   ├── leagues/
│   └── seasons/
├── viz/
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Entry Scripts

- `scripts/run_webapp.py`
- `scripts/train.py`
- `scripts/generate_projections.py`
- `scripts/fetch_player_insight_search.py`
- `scripts/synthesize_player_insights.py`
- `scripts/generate_position_guide.py`
