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
| `position-guide` | Generate static RL position probability cheat sheet | `python scripts/generate_position_guide.py --slot 5 --simulations 5000 --checkpoint-dir models/12_teams_random_start/v3` |

### Common Commands

Start the web application:

```bash
docker compose up webapp
```

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
   - `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/)
2. Optional: for Google CSE instead, set `INSIGHTS_SEARCH_PROVIDER=google`, `GOOGLE_CSE_API_KEY`, and `GOOGLE_CSE_ID` (note: CSE is closed to new customers and sunsets Jan 2027).

**Run order:**

```bash
# 1. Generate player projections (if not already done)
docker compose run --rm data

# 2. Fetch and cache search snippets (Valyu by default)
docker compose run --rm insights-search

# 3. Synthesize structured insights with Gemini Flash
docker compose run --rm insights-synthesize
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

### Position Guide (pre-draft cheat sheet)

Offline Monte Carlo simulation produces a **static position probability guide** for your draft slot — useful as a fallback when the live dashboard is unavailable.

**Prerequisites:**

1. Generate player projections (if not already done): `docker compose run --rm data`
2. A trained policy checkpoint (default: latest in `models/12_teams_random_start/v3/`)

**Run (12-team league, slot 5):**

```bash
docker compose run --rm position-guide
open data/guides/exports/position_guide_12teams_slot5_2026_*.html
```

**10-team league override:**

```bash
docker compose run --rm position-guide \
  python scripts/generate_position_guide.py --num-teams 10 --slot 5 \
  --checkpoint-dir models/10_teams_random_start/v1
open data/guides/exports/position_guide_10teams_slot5_2026_*.html
```

**Outputs:**

- JSON: `data/guides/exports/position_guide_{num_teams}teams_slot{slot}_{year}_{timestamp}.json`
- HTML: same basename with `.html` (printable cheat sheet)

Each run writes a new timestamped export. Filenames include league size so 12-team and 10-team guides do not collide.

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
