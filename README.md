# Draft Buddy - AI-Powered Fantasy Football Draft Assistant

Draft Buddy is a system for:

- training an RL draft agent,
- running an interactive web draft UI,
- generating projection data,
- and visualizing architecture dependencies.

The project has been refactored into bounded packages with explicit responsibilities (`core`, `data`, `simulator`, `rl`, `web`, `arch_viz`) and predictable script entry points in `scripts/`.

![ui](images/ui.png)
![loss](images/loss.png)
![reward](images/reward.png)

## Key Features

- Custom snake-draft environment with roster constraints (including FLEX handling).
- Opponent strategy system (`RANDOM`, `ADP`, `HEURISTIC`, `AGENT_MODEL`) with injection-based model inference.
- Feature engineering and reward shaping for RL training.
- FastAPI-backed web app with manual picks, simulation controls, CSV export, and AI suggestions.
- Stateless season evaluation pipeline for end-of-draft scoring.
- Architecture visualization CLI that emits Mermaid dependency graphs.

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Clone

```bash
git clone https://github.com/your-username/draft-buddy.git
cd draft-buddy
```

### Primary Runtime Modes (Docker Compose)

The repository standardizes around five operational services:

```bash
# Web app
docker compose up webapp

# RL training
docker compose run --rm train

# Test suite
docker compose run --rm test

# Test suite with branch coverage
docker compose run --rm test-cov

# Data preparation
docker compose run --rm data

# Architecture diagrams
docker compose run --rm ast
```

## Testing and Coverage

Run the fast test suite:

```bash
docker compose run --rm test
```

Run tests with branch coverage and artifact output:

```bash
docker compose run --rm test-cov
```

Coverage artifacts are written to:

- `coverage.xml` (CI/report tooling)
- `htmlcov/index.html` (local interactive report)

You can also run targeted package coverage directly, for example:

```bash
docker compose run --rm test python -m pytest tests/ \
  --cov=src/draft_buddy/web --cov-report=term-missing --cov-branch
```

## Web App

Start the server:

```bash
docker compose up webapp
```

Open the UI at `http://localhost:5001`.

UI capabilities include:

- start/reset draft session,
- manual picks and undo,
- single-step and rest-of-draft simulation,
- live roster breakdowns and player filtering,
- AI suggestions by team perspective,
- season simulation and CSV export.

## Training

Training entrypoint: `scripts/train.py`

1. Adjust config in `src/draft_buddy/config.py` (`TOTAL_EPISODES`, `LEARNING_RATE`, `ENABLED_STATE_FEATURES`, etc.).
2. Run training:

```bash
docker compose run --rm train
```

3. Resume by setting `RESUME_TRAINING = True` in config.
4. Generate plots without training:

```bash
docker compose run --rm train python scripts/train.py -p
```

## Data Preparation

Data entrypoint: `scripts/generate_projections.py`

Default compose run:

```bash
docker compose run --rm data
```

Custom year/options:

```bash
docker compose run --rm data python scripts/generate_projections.py --year 2024 --rookie_projection_method hybrid
```

## Architecture Diagrams (`draft-arch`)

The `arch_viz` CLI traces imports from entry files and emits Mermaid diagrams.

- Strategies: `module`, `class`, `function`
- Default entries:
  - `scripts/run_webapp.py`
  - `scripts/train.py`
  - `scripts/generate_projections.py`

Run with compose:

```bash
docker compose run --rm ast
```

Override strategy:

```bash
docker compose run --rm ast python -m draft_buddy.arch_viz.cli \
  --project-root /app --output-dir /app/viz --strategy class --all-default-entries
```

Custom entries:

```bash
docker compose run --rm ast python -m draft_buddy.arch_viz.cli \
  --project-root /app --output-dir /app/viz --strategy module \
  --entry scripts/run_webapp.py --entry scripts/train.py
```

Local usage (after `pip install -e .`):

```bash
python -m draft_buddy.arch_viz.cli --all-default-entries --output-dir viz --strategy module
draft-arch --entry scripts/run_webapp.py --strategy module
```

## Script Entry Points

The `scripts/` directory is intentionally minimal:

- `scripts/train.py`
- `scripts/run_webapp.py`
- `scripts/generate_projections.py`

## Package Structure

```text
.
├── data/                         # input/output datasets and artifacts
├── frontend/                     # browser UI assets
├── scripts/                      # canonical runtime entrypoints
├── src/draft_buddy/
│   ├── arch_viz/                 # dependency graph + Mermaid CLI
│   ├── core/                     # domain entities, draft state, rules, bots, inference abstraction
│   ├── data/                     # data adapters/loaders used by app/runtime layers
│   ├── data_pipeline/            # ETL and projection generation pipeline
│   ├── draft_env/                # gym environment adapter
│   ├── domain/                   # shared domain models
│   ├── rl/                       # policy network, training agent, checkpointing, rewards, feature extraction
│   ├── simulator/                # stateless schedule + season evaluator
│   └── web/                      # FastAPI app and session orchestration
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Notes on Legacy Modules

Some modules under `logic/` and `draft_env/` may remain as internal adapters while migration is in progress. New code should target `core`, `rl`, `simulator`, `data`, and `web` package boundaries directly.

## License

This project is released under the MIT License.
