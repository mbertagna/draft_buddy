# Draft Buddy

Draft Buddy uses Docker Compose as the primary local workflow for the refactored runtime boundaries:

- `web`: FastAPI app and session management
- `rl`: Gym environment, feature extraction, rewards, models, and training
- `data`: player loading and projection generation
- `simulator`: stateless season evaluation
- `core`: shared draft state, controller, rules, bots, and entities

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

Then recreate the webapp so Compose reloads `.env`:

```bash
docker compose up webapp
```

Each league keeps its own generated CSV, draft board, and saved states under `data/leagues/{league_id}/`. Run `docker compose run --rm data` **once per league** (first time, or after scoring/ADP/stats change). After that, flipping `DRAFT_BUDDY_LEAGUE` is enough for training, the UI, the data pipeline default year, and static guides. Insights search/synthesis caches stay shared across leagues.

If the generated CSV is missing, loaders fail instead of inventing dummy players. Pipeline scripts default `--year` from `DRAFT_BUDDY_SEASON`.

Build the image used by every service:

```bash
docker compose build
```

Each service bind-mounts the repository into `/app` and runs with `PYTHONPATH=/app/src`, so generated files are written back to your host checkout.

Common output locations on the host:

- `data/leagues/{league_id}/generated/{year}/`: player projections and pipeline report
- `data/leagues/{league_id}/runtime/{year}/`: live `draft_state.json` for that league/season
- `data/leagues/{league_id}/saved_states/`: archived draft boards
- `logs/`: training metrics, dashboards, and run logs
- `models/`: checkpoints and trained model artifacts
- `coverage.xml`: XML coverage report from `test-cov`
- `htmlcov/`: HTML coverage report from `test-cov`

### Services

| Service | Purpose | Default command |
| --- | --- | --- |
| `webapp` | Run the FastAPI web application | `python scripts/run_webapp.py` |
| `train` | Run RL training | `python scripts/train.py` |
| `trades` | In-season trade intelligence CLI | `python scripts/inseason_trades.py --help` |
| `test` | Run the test suite | `python -m pytest tests/` |
| `test-cov` | Run tests with coverage outputs | `python -m pytest tests/ --cov=src/draft_buddy ...` |
| `data` | Generate player projections and merged draft data | `python scripts/generate_projections.py --year 2026` |
| `insights-search` | Fetch web search snippets for top 150 ADP players (Tavily default) | `python scripts/fetch_player_insight_search.py --year 2026 --top-n 150` |
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

### In-season trades

Trade points are scored NFL weeks, not ADP. The two-year median, mean, floor, and ceiling come from 2024–2025 weekly fantasy points. `2026` points per game come from games already played this season. Rest-of-season totals blend those two. ADP is used when the catalog is built: rookies with no history get points by interpolating nearby veterans, and draft rankings use ADP. A veteran who already has weekly stats is not moved by ADP in trade search.

Run the steps in this order. `DRAFT_BUDDY_LEAGUE` picks the league. NBFL (`redraft_nbfl_12`) has a Sleeper id, so it pulls live rosters. Red League (`red_league_10`) does not, so you load a JSON snapshot. `config/inseason/manual_snapshot.example.json` is the shape to hand a chatbot when converting a screenshot. `--user` and `--partner` accept a team id or a manager name. `--send` and `--receive` accept a player id or a name.

With Docker Compose, pass the league with `-e` (or set it in `.env`). A shell prefix like `DRAFT_BUDDY_LEAGUE=red_league_10 docker compose run ...` does **not** override `.env` inside the container.

1. Build the player catalog. Repeat this when you want a new week of stats in the floor, median, mean, and ceiling.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 data
```

2. Load who owns whom. Sleeper leagues:

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 trades snapshot
```

Leagues without a Sleeper id:

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=red_league_10 trades load --file config/inseason/red_league_10_snapshot.json
```

3. Turn the catalog into remaining-season weekly points for the current rosters.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 trades projections
```

4. Search every other team for your best offers. The list is sorted by your playoff-odds gain. Each player line shows this year's points per game plus the two-year median, mean, floor, and ceiling.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 trades search --user Michael --csv data/leagues/redraft_nbfl_12/runtime/2026/trade_search.csv
```

Add `--partner "Erol"` to score deals with only that manager. Omit it to search every other team.

A schedule file is optional. Sleeper `snapshot` downloads the remaining slate and fills only weeks the API does not return. Other leagues keep a generated round-robin until you add a `matchups` list to the snapshot or run `schedule --file`. Home and away are a manager name or a team id. Weeks you leave out stay generated.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=red_league_10 trades schedule --file config/inseason/red_league_10_schedule.json
```

`--csv` writes every ranked trade to a file. Open that file when the terminal scrollback cuts the list off. The terminal still prints the same ranking. `send_1_*` is the first player you give up, `send_2_*` the second. `receive_1_*` and `receive_2_*` are the players you get. A blank slot means that side of the deal has only one player.

| Column | Meaning |
| --- | --- |
| `rank` | Order in the file. 1 is the best trade for you. |
| `partner` / `partner_team_id` | The other manager's name and team id. |
| `you_playoff_pp` / `them_playoff_pp` | Change in playoff odds, in percentage points, averaged over the season sims. `2.0` means 2 points more likely to make the playoffs, not 2% of the previous chance. |
| `you_expected_wins_delta` / `them_expected_wins_delta` | Change in expected season wins, including the record already locked. |
| `you_starter_ros_delta` / `them_starter_ros_delta` | Change in projected points from the best lineup over the rest of the regular season. This is calculated before the season sims. |
| `*_player_id` / `*_position` / `*_name` | Who is in that slot. |
| `*_2026_ppg` | Points per game already scored this season. Blank when they have no 2026 games. |
| `*_median` / `*_mean` | Typical and average weekly points from the 2024–2025 seasons. |
| `*_floor` / `*_ceiling` | A bad week and a big week from those two seasons (10th and 90th percentile), in points per game. |

A trade is good for you when `you_playoff_pp` is at least 2 and `you_expected_wins_delta` is above 0. It is good for both managers when all four `you_*` and `them_*` playoff and wins columns are above 0. Both can gain when the deal helps you pass teams who are not part of the trade.

Add `--exhaustive` to score every 1-for-1, 2-for-1, 1-for-2, and 2-for-2 deal between your skill players and each other roster. That mode skips the top-player pool and the candidate cap, then sorts by your playoff gain. Every offer still uses 200 season sims. Those sims run on every CPU by default; `--workers 4` sets the process count and `--workers 1` stays on one process. The terminal prints the first 20; the CSV has the rest.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=red_league_10 trades search --user "California Fourskin" --exhaustive --csv data/leagues/red_league_10/runtime/2026/trade_search.csv
```

5. Score one deal you already have in mind.

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 trades evaluate --user Michael --partner Erol --send "Drake London" --receive "Ja'Marr Chase"
```

A full search explores all nine opponents, but only the top 60 heuristic candidates get Monte Carlo simulation (`trade.max_monte_carlo_candidates` in the league JSON). When one matchup dominates that pre-filter — for example, sell-high QBs on your bench against a team with elite WR/RB starters — every printed offer can be with the same partner. Use `evaluate --partner <name>` to score a specific deal with someone else. Each surviving offer is simulated at 200 runs, plus one baseline season, about 12,200 remaining-season sims total. Every simulated deal is kept and sorted by your playoff-odds gain, including ones where the partner loses. `show` prints the current snapshot and rest-of-season totals without searching. Add `--json` to `search` or `evaluate` for the same numbers as JSON.

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
docker compose run --rm data python scripts/generate_projections.py --year 2024
```

Rookie points are estimated by ADP neighbor interpolation among same-position veterans, with NFL draft-slot scaling as a fallback when ADP matching leaves gaps.

### FantasyPros ADP (manual HTML snapshot)

Projection generation merges FantasyPros consensus ADP from a saved HTML table (not the CSV export).

1. Open [FantasyPros PPR Overall ADP](https://www.fantasypros.com/nfl/adp/ppr-overall.php)
2. In DevTools inspector, select the ADP results **`<table>`** whose classes include `mcu-table` / `reports__table-inner` (caption text starts with `Average Draft Position (ADP)`)
3. Copy that table element's **outer HTML** only (not the full page, and not an ECR rankings table)
4. Save it as `data/cache/adp/fantasypros-{year}-overall-adp-rankings.html` (example: `fantasypros-2026-overall-adp-rankings.html`)
5. Run `docker compose run --rm data`

Do **not** use the ECR / rankings page table. The ADP table must include an `AVG` column.

### Player Insights (manual pre-draft enrichment)

Offline player insight enrichment is a **manual, two-step** pipeline that prepares research-backed outlook data for the draft UI (see [PLAYER_INSIGHTS_PART2_PLAN.md](docs/PLAYER_INSIGHTS_PART2_PLAN.md)).

**Prerequisites:**

1. Copy `.env.example` to `.env` and set:
   - `TAVILY_API_KEY` from [Tavily](https://www.tavily.com/) (default search provider; 1,000 free credits/month, basic search)
   - Optional `VALYU_API_KEY` from [Valyu](https://platform.valyu.ai/) if you want the Valyu backup
   - `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/) and/or `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai/)

**Run order:**

```bash
# 1. Generate player projections (if not already done)
docker compose run --rm data

# 2. Fetch and cache search snippets (Tavily; date-filtered + relevance >= 0.7)
docker compose run --rm insights-search

# 3. Synthesize structured insights (Gemini or OpenRouter)
docker compose run --rm insights-synthesize
```

Tavily is the default search provider (`INSIGHTS_SEARCH_PROVIDER=tavily`). Search uses Tavily basic depth (1 credit per query), publication date windows (`outlook`/`role`: March 1 of the draft year; `injury_recovery`: last 90 days), and a `0.7` relevance threshold. Injury queries use Tavily `topic=news`; outlook/role use `general`. Synthesis then re-ranks cached snippets by relevance and recency before calling the LLM.

To use Valyu instead:

```bash
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --provider valyu
```

Or set `INSIGHTS_SEARCH_PROVIDER=valyu` in `.env`.

After upgrading the insights pipeline, **re-fetch search caches** so results include relevance scores and date windows:

```bash
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --force
docker compose run --rm insights-synthesize python scripts/synthesize_player_insights.py --force
```

**Synthesis model selection:**

Defaults come from `.env` (`INSIGHTS_LLM_PROVIDER`, `INSIGHTS_LLM_MODEL`). Override per run:

```bash
docker compose run --rm insights-synthesize python scripts/synthesize_player_insights.py \
  --provider openrouter --model deepseek/deepseek-v4-flash
```

**Outputs:**

- Search cache runs: `data/cache/insights/search/runs/{timestamp}/` with per-player dirs
- Synthesis cache runs: `data/cache/insights/synthesis/runs/{timestamp}/{sleeper_id}.json`
- Active run pointer: `current.json` (`{"run_id": "..."}`) under each of those cache roots
- Merged insights export: `data/insights/exports/player_insights_{year}_{timestamp}.json`

Legacy flat layouts (`search/{sleeper_id}/…`, `synthesis/{sleeper_id}.json`) are still read when `current.json` is missing.

Without `--force`, new players are written into the **current** run (or the legacy flat root). With `--force`, a **new empty** timestamped run is created and becomes current; prior runs are left untouched. A partial `--force` (for example `--max-players 20`) makes that incomplete run the new current set.

Each synthesis run also writes a new timestamped export file. The webapp loads the newest export by filename timestamp at startup. Legacy undated `data/player_insights_{year}.json` files are used as a fallback when no exports exist yet.

To restore an older UI insights snapshot, keep or restore the desired file as the newest under `data/insights/exports/` (rename/move aside newer exports), then restart the webapp.

**Partial re-runs:**

```bash
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --max-players 20 --start-index 0
docker compose run --rm insights-search python scripts/fetch_player_insight_search.py --force
docker compose run --rm insights-synthesize python scripts/synthesize_player_insights.py --force
```

`--force` starts a fresh empty run; prefer a full top-N force refresh unless you intentionally want a smaller current set.
### Live Draft Assistant

The web UI includes an on-demand **LLM draft assistant** alongside the fast RL position chips. Set `GEMINI_API_KEY` and/or `OPENROUTER_API_KEY` in `.env`.

**Supported models:** Gemini 2.5 Flash, Gemini 2.5 Flash Lite, DeepSeek V4 Pro, DeepSeek V4 Flash (via OpenRouter).

**In the header:**

- **Auto assistant** — when on, fires once per snake turn when scope allows (skipped during clock overrides)
- **Scope** — *My picks only* (agent team from league config) or *Every team* (auto only)
- **My model / Others** — separate model pickers for your team vs other teams (defaults from `ADVISOR_AGENT_MODEL` / `ADVISOR_OTHER_TEAMS_MODEL`)
- **Ask Assistant** — always available during an active draft; uses the selected/on-clock team (including overrides)

The assistant builds per-position shortlists (top 7 by VORP/ADP for the RL model's top two positions, top 5 for the others) and returns a structured pick recommendation. Min GP Frac from the player table is sent with each request.

### Sleeper live draft sync

Draft Buddy can **mirror a live Sleeper snake draft** instead of running a local mock. You do **not** need a Sleeper API key. The public `https://api.sleeper.app/v1` endpoints used here are unauthenticated. Gemini/OpenRouter keys are still only for the LLM assistant.

This is a companion mode: every pick, including yours, comes from Sleeper. Pick, undo, transfer, swap, clock override, Sim Pick, and Auto Draft are disabled. Advisor, RL chips, the player table, shelve, and the board stay on. Your team is `draft.AGENT_START_POSITION` (slot 5 / Michael for Redraft NBFL).

**Enable it in season JSON**, not `.env`. For Redraft NBFL edit `config/seasons/redraft_nbfl_12_2026.json`:

```json
"draft": {
  "AGENT_START_POSITION": 5,
  "SLEEPER_SYNC_ENABLED": true,
  "SLEEPER_DRAFT_ID": "your_draft_id",
  "SLEEPER_LEAGUE_ID": "your_league_id",
  "SLEEPER_ROSTER_ID_TO_TEAM_ID": {
    "1": 5,
    "2": 1
  },
  "SLEEPER_SYNC_POLL_SECONDS": 3
}
```

`SLEEPER_ROSTER_ID_TO_TEAM_ID` maps each Sleeper `roster_id` onto Draft Buddy team ids `1..NUM_TEAMS` (the same keys as `TEAM_MANAGER_MAPPING`). The example values above are placeholders — paste a full 1:1 map for all 12 teams. Leave `SLEEPER_SYNC_ENABLED` false (and ids empty) for local mocks. ESPN `red_league_10` stays unsynced.

**One-time lookup** (league id is in the Sleeper league URL):

```bash
curl -s "https://api.sleeper.app/v1/league/{league_id}/drafts"   # copy draft_id
curl -s "https://api.sleeper.app/v1/league/{league_id}/rosters" # roster_id + owner_id
curl -s "https://api.sleeper.app/v1/league/{league_id}/users"   # owner_id → display_name
```

Match each Sleeper owner to `TEAM_MANAGER_MAPPING`, then set `DRAFT_BUDDY_LEAGUE=redraft_nbfl_12` and restart:

```bash
docker compose up webapp
```

Startup fails loud if the draft id is missing, the roster map is not 1:1 onto `1..NUM_TEAMS`, or Sleeper `type` is not `snake`. Restart after flipping the flag or changing ids. **New Draft** replays that Sleeper draft from pick 0 (it does not start a local mock).

While syncing, the UI polls Draft Buddy every `SLEEPER_SYNC_POLL_SECONDS` (default 3). Polling pauses when the tab is hidden or Sleeper status is `complete`. Kickers and DST show on extra board rounds with a **No proj** badge and do not count toward skill roster slots, VORP, advisor, or RL. Off-catalog skill players appear as name/position placeholders. If Sleeper drafts a player you shelved locally, Draft Buddy unshelves them and applies the pick.

### Shelved players

Players can be removed from the draftable pool without drafting them onto a team:

- Per-row **−** control in the player table, **Hide ADP above**, or **Shelve inactive** (Inactive/IR/PUP/DNR) in the header **Shelved** panel
- Open **Shelved** in the banner (same pattern as Settings/Stats) for ADP-sorted chips; click a chip or **Restore all** to unshelve
- They are excluded from VORP, bots, Sim/Auto Draft, RL chips, and the LLM assistant
- Persisted in draft state JSON as `shelved_player_ids` (not on the visual board or pick history)
- New Draft does **not** auto-shelve by default (`data.AUTO_SHELVE_INACTIVE_ON_NEW_DRAFT`); use **Shelve inactive** when you want that filter
### Training draft pool regimes

When `training.RANDOMIZE_DRAFT_POOL_DURING_TRAINING` is true, each training episode samples a weighted regime from `training.DRAFT_POOL_REGIMES`:

| Regime id | Behavior |
| --- | --- |
| `full` | Entire catalog |
| `inactive_only` | Drop Inactive/IR/PUP/DNR |
| `adp_only` | Top ADP pool of size `num_teams * roster_slots + U(EXTRA_MIN, EXTRA_MAX)` |
| `inactive_and_adp` | Inactive prune then ADP top-N |

Defaults use `ADP_POOL_EXTRA_MIN=40` / `MAX=80` so tight ADP boards keep slack beyond a full draft. ADP-limited pools also top up by position so RB/WR/TE supply covers worst-case flex absorption (bench caps can otherwise exhaust a position while a team still has roster room). Season overlays live under `config/seasons/`. With randomization on, the gym keeps the full catalog so full-pool episodes remain possible. If an opponent still cannot pick, training ends that episode early and prints a detailed `WARNING` (regime, pool size, roster counts, available positions) instead of crashing.

See [PLAYER_INSIGHTS_PART2_PLAN.md](docs/PLAYER_INSIGHTS_PART2_PLAN.md) for architecture details.

### Position Guide (pre-draft cheat sheet)

Offline Monte Carlo simulation produces **static position probability guides for every draft slot in one run** — useful as a fallback when the live dashboard is unavailable. Every team samples its own temperature-scaled policy suggestion each pick ("all-slots self-play"), so one batch of simulations yields self-consistent guides for all slots simultaneously, plus a league-wide model-derived ADP ranking. Defaults (`num-teams`, checkpoint) come from the active league profile in `.env`.

**Prerequisites:**

1. Generate player projections for the active league: `docker compose run --rm data`
2. A trained policy checkpoint for that league size, pointed at by `training.MODEL_PATH_TO_LOAD` in `config/seasons/{league}_{year}.json` (a checkpoint file or a directory containing `checkpoint_episode_*.pth` files — the latest episode is used)

**Run (uses active league from `.env`):**

```bash
docker compose run --rm position-guide
open data/guides/exports/position_guide_*_*teams_slot*_*_*.html
open data/guides/exports/model_adp_*_*teams_*_*.html
```

**Override league temporarily:**

```bash
docker compose run --rm -e DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 position-guide
```

**Useful flags:**

- `--temperature` (default `1.5`) — softmax temperature applied to every self-play suggestion; values above `1.0` soften an overconfident policy (e.g. ~99%/1% QB/RB splits) while preserving its ranking
- `--prune-inactive` — exclude Inactive/IR/PUP/DNR players from this run's draftable pool (same statuses as UI **Shelve inactive**)
- `--limit-adp N` — keep the top `N` players by ascending FantasyPros ADP, then top up to worst-case position floors (final pool may exceed `N`); omit for the full catalog
- `--checkpoint` — override the configured checkpoint file or directory

Example with both pool filters:

```bash
docker compose run --rm position-guide python scripts/generate_position_guide.py --simulations 5000 --prune-inactive --limit-adp 200
```

**Outputs:**

- Per-slot guide JSON: `data/guides/exports/position_guide_{league_id}_{num_teams}teams_slot{slot}_{year}_{timestamp}.json`
- Per-slot guide HTML: same basename with `.html` (printable cheat sheet, with top-5 actually-drafted players shown under each position bar)
- Model ADP JSON: `data/guides/exports/model_adp_{league_id}_{num_teams}teams_{year}_{timestamp}.json`
- Model ADP HTML: same basename with `.html` (players ranked by mean simulated overall pick number, alongside market ADP)

Each run writes new timestamped exports for every slot plus one model-ADP export. Filenames include league id and league size so 10-team and 12-team (or two same-size leagues) guides do not collide.

### League profiles and season rollover

| League | Profile ID | Platform | Scoring | 2026 draft slot |
| --- | --- | --- | --- | --- |
| Red League | `red_league_10` | ESPN | Full PPR | 2 |
| Redraft NBFL | `redraft_nbfl_12` | Sleeper | Half PPR | 5 |

**Season rollover checklist** (each August):

1. Copy `config/seasons/{league}_2026.json` to `{league}_2027.json`
2. Update `season`, `bye_weeks`, and `draft.AGENT_START_POSITION`
3. Run `docker compose run --rm data` for each league you use
4. Train or point `training.MODEL_PATH_TO_LOAD` at the correct `models/{N}_teams_*` checkpoint (file or directory) — this is the single source of truth for both the webapp and the position guide

Player projections only include nflverse-trackable scoring rules. Bonuses without reliable stat columns (50+ yard TDs, D/ST details, IR slots) are omitted per league JSON.

### `up` vs `run --rm`

Use `docker compose up` for long-running services that should stay attached to a port, such as `webapp`.

Use `docker compose run --rm` for one-off tasks such as training, tests, coverage, and data generation. The `--rm` flag removes the container when the command exits.

Because every service uses `working_dir: /app`, command overrides run from the repository root inside the container. That means overrides like:

```bash
docker compose run --rm test python -m pytest tests/test_config.py
```

behave consistently across services.

### Accessing Outputs

- Web UI: [http://localhost:5001](http://localhost:5001)
- Coverage HTML report: [htmlcov/index.html](htmlcov/index.html)
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
│   ├── core/
│   ├── data/
│   ├── rl/
│   ├── simulator/
│   └── web/
├── config/
│   ├── leagues/
│   └── seasons/
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Entry Scripts

- `scripts/run_webapp.py`
- `scripts/train.py`
- `scripts/inseason_trades.py`
- `scripts/generate_projections.py`
- `scripts/fetch_player_insight_search.py`
- `scripts/synthesize_player_insights.py`
- `scripts/generate_position_guide.py`
