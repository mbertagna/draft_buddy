# Product Requirements Document (PRD)

## Project Title: In-Season Trade Intelligence (`draft_buddy` v1)

---

## 1. Executive Summary

In-season trade intelligence extends Draft Buddy from pre-draft optimization into remaining-season roster decisions. v1 answers two questions:

1. Should I accept this offer?
2. Which 1- and 2-player trades should I send?

It does **not** replace the draft session. Draft state and in-season state stay separate. Both supported leagues share one engine, accessed as a Compose CLI (`trades`) like `train`. Sleeper leagues pull live rosters when `SLEEPER_LEAGUE_ID` is set; non-Sleeper leagues load a JSON snapshot file (typically generated from a screenshot).

### v1 user stories

- As a manager, I want to evaluate an inbound N-for-M offer with smart bench drops and remaining-week playoff odds so I can accept, reject, or counter with numbers.
- As a manager, I want a ranked list of 1v1 / 2v1 / 1v2 / 2v2 offers so I can pitch trades that help me without being implausible for the other side.

### Out of v1

- FAAB / waiver assistant
- 3-player (or larger) outbound search
- ESPN API sync
- Full-league insight refresh
- LLM as the ACCEPT / REJECT / COUNTER decision-maker
- Championship-odds ranking

---

## 2. System Architecture

```
Sleeper adapter ──┐
                  ├── LeagueSnapshot ──┬── InboundTradeService ──┐
Manual JSON ──────┘                    │                         ├── SmartDropEvaluator
                                       └── TradeFinderService ───┘
                                                    │
                                       SlidingScaleProjectionEngine
                                                    │
                                       RemainingWeekMonteCarlo
                                                    │
                                       simulate_season_fast (FLEX from config)
```

---

## 3. Design Locks

### Primary metric

Remaining-season **playoff probability**, with expected wins shown alongside.

### Monte Carlo

Sample remaining weekly points as `max(0, N(mu, 0.30 * mu))`. Bye weeks stay 0. Lock current W-L-T. Simulate weeks `current_week .. NUM_REGULAR_SEASON_WEEKS` plus playoffs.

- Inbound eval: 400 runs
- Outbound search: 200 runs per surviving candidate (cap 30)

This is a wrapper around `simulate_season_fast`, not a loop on the existing deterministic `SeasonSimulationService` UI path.

### Verdict thresholds (numeric)

| Verdict | User ΔP(playoff) |
| --- | --- |
| ACCEPT | ≥ +0.02 |
| COUNTER | ≥ -0.01 |
| REJECT | < -0.01 |

Optional LLM pitch / defensive blurb runs **after** the numbers exist and never overrides the verdict.

### Outbound fairness

Keep a candidate only if partner ΔP(playoff) ≥ -0.01 **or** partner Δ expected wins ≥ -0.05. Optimize the user's odds, not strict Pareto starter points.

### Outbound shapes

`1v1`, `2v1`, `1v2`, `2v2`. Inbound eval accepts any N-for-M.

### Platforms

Shared `LeagueSnapshot` plus two adapters. Sleeper vs file is determined by the **active league profile** (`DRAFT_BUDDY_LEAGUE`), not a second service. Unmatched team or player names are errors; the engine does not guess.

---

## 4. Feature Specifications

### Feature 1: League snapshot

`LeagueSnapshot` holds current week, team rosters, IR ids, W-L-T, and remaining matchups.

- **Sleeper:** `fetch_nfl_state`, rich roster payload (`players`, `reserve`, wins/losses/ties), per-week matchups. Map `roster_id` with `SLEEPER_ROSTER_ID_TO_TEAM_ID`.
- **Manual / ESPN file:** `{current_week, teams: [{team_id|team|display_name, wins, losses, player_names|player_ids}]}`. Match manager names against `TEAM_MANAGER_MAPPING` and player names with `standardize_name`. Reject unmatched or ambiguous names. Example: `config/inseason/manual_snapshot.example.json`.
- Persist under `data/leagues/{league_id}/runtime/{year}/inseason_state.json`.

Schedule fallback: Sleeper remaining matchups when present; otherwise round-robin for leftover weeks. Past weeks are not re-simulated.

### Feature 2: Sliding-scale ROS projections

Weekly ROS vectors, not a single blended season total.

```
ppg = (W_hist * points_hist) + (W_curr * points_curr)
```

Default weights:

- Weeks 1–3: `W_hist = 0.70`, `W_curr = 0.30`
- Weeks 4–6: `W_hist = 0.40`, `W_curr = 0.60`
- Weeks 7+: `W_hist = 0.10`, `W_curr = 0.90`

Rules:

- `points_hist` is per-game median from lookback seasons.
- `points_curr` is per-game mean of completed current-season weeks.
- Rookies use the preseason / rookie-projector value as `points_hist`.
- If current-season games played < 3, cap `W_curr` at the weeks 1–3 weight.
- RB listed as depth-chart order 1 with fewer than 3 games as that starter: raise `W_curr` but cap at 0.70.
- Emit `{player_id: {pos, pts: [18]}}` with bye weeks zeroed from the season overlay / player bye.

`docker compose run --rm trades projections` (or `scripts/refresh_inseason_projections.py`) rewrites ROS vectors after nflverse pulls. `PlayerCatalog.to_weekly_projections` also zeros byes so `/api/simulate_season` matches.

### Feature 3: Remaining-week simulator

`simulate_season_fast` lineups follow `config.draft.ROSTER_STRUCTURE` (FLEX 3 in both leagues). It accepts `start_week` and `starting_records`. `RemainingWeekMonteCarlo` adds noise and aggregates `p_playoff` and `expected_wins`. The existing Sim Season button stays deterministic.

### Feature 4: Smart drops + inbound eval

If a trade exceeds `starters + TOTAL_BENCH_SIZE`, generate up to `trade.smart_drop_variations_count` (default 3) drop sets from the lowest ROS bench (IR excluded). Rank drop sets with one deterministic remaining-week sim; Monte Carlo the winner.

`InboundTradeService` applies the swap, chooses drops, compares Monte Carlo vs baseline, and attaches ROS starter-point delta from `calculate_roster_scores`.

### Feature 5: Outbound finder

1. Macro needs from `categorize_roster_by_slots` on ROS points.
2. Per-team pool: top 8 ROS plus positional-hole fillers; drop players below `ros_point_floor`.
3. Generate only allowed shapes vs each opponent.
4. Apply smart drops. Keep if user starter ROS delta > 0 **or** a hole improves, and partner starter ROS delta ≥ -3 points. Cap 30.
5. Monte Carlo those 30. Rank by user ΔP(playoff) with the partner fairness floor.
6. Optional LLM pitch for the top 3 only.

---

## 5. Configuration

Exposed via `config/leagues/<league_id>.json` (and season overlays). Defaults:

| Key | Default | Description |
| --- | --- | --- |
| `inseason.monte_carlo_runs_eval` | `400` | MC runs for inbound eval |
| `inseason.monte_carlo_runs_search` | `200` | MC runs per search candidate |
| `inseason.monte_carlo_noise_sigma` | `0.30` | Weekly noise as a fraction of mu |
| `trade.max_players_per_side_outbound` | `2` | Max players per side in search |
| `trade.outbound_shapes` | `[[1,1],[2,1],[1,2],[2,2]]` | Allowed search shapes |
| `trade.max_monte_carlo_candidates` | `60` | Heuristic survivors sent to MC |
| `trade.smart_drop_variations_count` | `3` | Drop permutations scored |
| `trade.partner_playoff_delta_floor` | `-0.05` | Partner ΔP(playoff) floor |
| `trade.partner_expected_wins_floor` | `-0.05` | Alternate partner fairness floor |
| `trade.partner_starter_delta_epsilon` | `-8.0` | Heuristic partner ROS-point floor |
| `trade.player_pool_top_n` | `8` | Top ROS players per team in search |
| `trade.ros_point_floor` | `20.0` | Exclude low-ROS bench from search |
| `trade.accept_playoff_delta` | `0.02` | ACCEPT threshold |
| `trade.counter_playoff_delta` | `-0.01` | COUNTER threshold |
| `projections.sliding_scale_weights` | *see 4.2* | Week-range hist/curr weights |
| `projections.min_current_games_for_full_weight` | `3` | Low-sample cap |
| `projections.depth_chart_curr_weight_cap` | `0.70` | Promoted-RB `W_curr` cap |
| `projections.depth_chart_starter_games` | `3` | Games before uncapping RB bump |

---

## 6. Compose CLI

Service `trades` in `docker-compose.yml` (`entrypoint: python scripts/inseason_trades.py`). League selection is `DRAFT_BUDDY_LEAGUE`.

```bash
DRAFT_BUDDY_LEAGUE=redraft_nbfl_12 docker compose run --rm trades snapshot
DRAFT_BUDDY_LEAGUE=red_league_10 docker compose run --rm trades load --file config/inseason/manual_snapshot.example.json
docker compose run --rm trades projections
docker compose run --rm trades show
docker compose run --rm trades evaluate --user Michael --partner Joey --send "Jahmyr Gibbs" --receive "Bijan Robinson"
docker compose run --rm trades search --user 5
```

`snapshot` requires `draft.SLEEPER_LEAGUE_ID`. Otherwise use `load --file`. `--user` / `--partner` accept team id or manager name; `--send` / `--receive` accept player ids or names. Add `--json` on a subcommand for machine-readable output.

Prompt for a screenshot dump: produce JSON matching `config/inseason/manual_snapshot.example.json` (`current_week`, each team’s `team` name from the league mapping, W-L, and `player_names`). Do not invent unmatched names.

Draft webapp flows are unchanged; there is no Trade panel.

---

## 7. Implementation Phases

1. Snapshot + adapters + config
2. ROS weekly vectors
3. Simulator FLEX / remaining-week / Monte Carlo
4. Smart drops + inbound eval
5. Constrained outbound finder
6. Compose CLI (`trades`)
