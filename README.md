# Draft Buddy – RL-powered Fantasy Draft Assistant

Draft Buddy is a complete system for simulating, training, and running a fantasy football draft assistant. It combines a custom OpenAI Gym environment, a REINFORCE policy network, opponent strategy personalities, and a simple web UI to let you run mock drafts, visualize rosters, and train an agent to draft intelligently.

---

## Key Features

- Custom Gym environment modeling a multi-team snake draft with realistic roster rules
- Rich observation space (VORP, scarcity, top-k points, opponent threat, bye weeks, etc.)
- Pluggable opponent strategies: HEURISTIC, ADP, RANDOM, and optional agent-model opponents
- Optional opponent randomization per episode for training diversity
- Configurable reward function with season simulation bonuses and per-pick shaping
- Flask backend + lightweight frontend to run manual/sim drafts with AI suggestions
- Save/Load draft state, manual override per pick, undo, CSV export
- Fast season simulation (parallelized subprocess) using weekly points-by-position rules
- Training utilities with logging and plotting; checkpoint resume support

---

## Project Structure

```
.
├── app.py                         # Flask API and static file server
├── fantasy_draft_env.py           # Gym environment (core draft logic + rewards)
├── policy_network.py              # Policy network used by agent/opponents
├── reinforce_agent.py             # REINFORCE training loop
├── train.py                       # CLI to train agent and plot latest logs
├── simulate.py                    # Simulate drafts using a trained policy
├── utils/
│   ├── season_simulation_utils.py # Season sim orchestration & helpers
│   └── fast_solve_matchups.py     # Parallel solver for weekly matchups
├── data_utils.py                  # Player data loading and structures
├── config.py                      # Central configuration
├── frontend/
│   └── index.html                 # UI (served by Flask)
├── data/                          # CSV inputs (players, matchups) and saved draft state
└── logs/, models/                 # Training outputs
```

---

## Requirements & Installation

- Docker installed on your system
- Recommended: Docker Desktop for easier management

```bash
# Build the Docker image
./build.sh

# Run the Docker container (interactive shell with mounted volume)
./run.sh
```

The Docker setup automatically handles all Python dependencies and provides an isolated environment. Your project directory is mounted at `/app` in the container, so changes are reflected immediately.

---

## Data

Place player and matchup CSVs in `data/`:

- `generated_player_data.csv` (required): columns include `player_id, name, position, projected_points, [adp], [games_played_frac], [bye_week], [team]`.
  - If `adp` is missing, a mock ADP is generated per `Config.MOCK_ADP_CONFIG`.
- `red_league_matchups_2025.csv` (optional): enables season simulation rewards.

A saved draft state is written to `data/draft_state.json` by the UI.

---

## Running the Web App (UI)

```
python app.py
# open http://localhost:8000
```

UI highlights:
- Header shows current pick number, team on the clock, and AI suggestion for that team.
- Team buttons show the snake order; the current team is highlighted. Full teams are visibly styled.
- Controls: Start New Draft, Undo, Simulate Pick, CSV Export.
- Player list can be filtered by position, searched by name, and sorted (e.g., by VORP).
- Manual override: select a team to pick next (useful even after the scheduled snake draft is over to fill rosters).

---

## API Endpoints (selected)

- `GET /api/draft/state` – full state for UI
- `POST /api/draft/new` – reset to a fresh draft
- `POST /api/draft/pick` – draft a player for the current team (JSON: `{ player_id }`)
- `POST /api/draft/undo` – undo last pick
- `POST /api/draft/simulate_pick` – simulate one pick for team on clock
- `POST /api/draft/override_team` – set the next picking team (JSON: `{ team_id }`)
- `GET /api/draft/ai_suggestion` – AI suggestion for team on clock
- `GET /api/draft/ai_suggestion_for_team?team_id=ID` – AI suggestion from a specific team’s perspective
- `GET /api/players?position=WR,TE&search=...&sort_by=vorp&sort_dir=desc` – player catalog with VORP sorting

---

## The Draft Environment

- Action space: 4 discrete actions → pick by position: QB, RB, WR, TE
- Draft order: N-team snake (generated per `ROSTER_STRUCTURE` and `TOTAL_BENCH_SIZE`)
- Roster rules:
  - Starters: QB, RB×2, WR×2, TE×1, FLEX×3 (RB/WR/TE)
  - Positional bench caps and total bench size enforced for AI; manual picks allow bench-only constraint to keep UI flexible
- Opponents:
  - `HEURISTIC`, `ADP`, `RANDOM`, or `AGENT_MODEL`
  - Optional per-episode randomization from `Config.OPPONENT_STRATEGY_TEMPLATES` for training diversity

### Observation space (selected)
- Best-available points per position, VORP per position
- Current roster counts and available slots
- Draft context: current pick number, agent start position
- Scarcity, top-k points, imminent opponent threat
- Full bye-week vector (weeks 4–14)

---

## Rewards

Terminal reward (always included):
- Roster-slot weighted points (starters × `STARTER_POINTS_WEIGHT`, bench × `BENCH_POINTS_WEIGHT`)
- Optional season simulation bonus (`SEASON_SIM_REWARDS`) using `red_league_matchups_2025.csv`

Per-pick shaping (small, to accelerate learning):
- Starter lineup delta: reward += max(0, Δ starter points) × `PICK_SHAPING_STARTER_DELTA_WEIGHT`
- VORP shaping: reward += max(0, drafted_points − baseline) × `VORP_PICK_SHAPING_WEIGHT`

Other knobs:
- Competitive difference modes (max or average opponent) – distinct from season sim mode
- Invalid action penalties (disabled by default)

All knobs are in `config.py`.

---

## Training

```
python train.py               # full training run per current Config
python train.py -plc          # only plot latest logs (auto-discovers latest logs dir)
```

Training outputs:
- Models saved under `models/<run_name>/v<version>/`
- Logs (raw CSVs and plots) under `logs/<run_name>/v<version>/`

Tips:
- If your real draft slot is fixed (e.g., pick 10), pretrain with a mix of slots (biased to 10), then fine-tune at slot 10.
- If the policy over-picks TE early, reduce `VORP_PICK_SHAPING_WEIGHT` (e.g., 0.03) and/or remove `te_scarcity` from enabled features.

---

## Simulation of Drafts (Evaluation)

Use a trained model to run multiple complete mock drafts and print detailed logs:

```
python simulate.py
```

This reports pick-by-pick logs, final roster points per team, and averages across runs.

---

## Configuration Highlights (`config.py`)

- Team & draft:
  - `NUM_TEAMS`, `AGENT_START_POSITION`
  - `ROSTER_STRUCTURE`, `BENCH_MAXES`, `TOTAL_BENCH_SIZE`
- Opponents:
  - `OPPONENT_TEAM_STRATEGIES` (+ `DEFAULT_OPPONENT_STRATEGY`)
  - Randomization: `RANDOMIZE_OPPONENT_STRATEGIES`, `RANDOMIZE_ONLY_DURING_TRAINING`, `OPPONENT_STRATEGY_TEMPLATES`
- State features: `ALL_STATE_FEATURES`, `ENABLED_STATE_FEATURES`, `STATE_NORMALIZATION_METHOD`
- Rewards:
  - `ENABLE_ROSTER_SLOT_WEIGHTED_REWARD`, `STARTER_POINTS_WEIGHT`, `BENCH_POINTS_WEIGHT`
  - `ENABLE_SEASON_SIM_REWARD`, `SEASON_SIM_REWARDS`
  - `ENABLE_PICK_SHAPING_REWARD`, `PICK_SHAPING_STARTER_DELTA_WEIGHT`
  - `ENABLE_VORP_PICK_SHAPING`, `VORP_PICK_SHAPING_WEIGHT`
  - Competitive: `ENABLE_COMPETITIVE_REWARD`, `COMPETITIVE_REWARD_MODE`
- Training: `TOTAL_EPISODES`, `LEARNING_RATE`, `DISCOUNT_FACTOR`, `HIDDEN_DIM`

---

## Troubleshooting

- Gym/Gymnasium API mismatch: the env uses the newer `(obs, reward, done, truncated, info)` return signature; ensure your installed Gym version matches requirements.
- Torch model device: models are loaded with `map_location='cpu'` to avoid CUDA issues on CPU.
- “AI model not loaded”: set `MODEL_PATH_TO_LOAD` to a valid `.pth` file or disable AI suggestions.
- Season sim: ensure `data/red_league_matchups_2025.csv` exists (or disable season sim rewards).

---

## License

MIT License (see `LICENSE`).

---

## Acknowledgments

- Thanks to open-source contributors behind Gym, PyTorch, and Flask.
- Player projections and matchup logic should be adapted to your league’s data.