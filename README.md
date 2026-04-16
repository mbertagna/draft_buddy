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
| `data` | Generate player projections and merged draft data | `python scripts/generate_projections.py --year 2025` |
| `ast` | Generate Mermaid architecture diagrams | `python -m draft_buddy.arch_viz.cli --project-root /app --output-dir /app/viz --all-default-entries --strategy module` |

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

Override the data-generation command:

```bash
docker compose run --rm data python scripts/generate_projections.py --year 2024 --rookie_projection_method hybrid
```

Generate architecture diagrams:

```bash
docker compose run --rm ast
```

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
