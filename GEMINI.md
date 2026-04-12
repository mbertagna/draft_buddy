# Draft Buddy - GEMINI Context

This project is an AI-powered Fantasy Football Draft Assistant that leverages reinforcement learning to train an agent for optimal drafting. It uses a custom OpenAI Gym environment to simulate snake drafts and evaluates performance through season simulations.

## Project Overview

- **Core Technology:** Python 3.10+, PyTorch (REINFORCE algorithm), OpenAI Gym, Flask, Pandas, NumPy.
- **Architecture:**
  - `api/`: Flask backend providing draft state, suggestions, and simulation endpoints.
  - `frontend/`: Vanilla JS/HTML/CSS UI for interactive drafting.
  - `src/draft_buddy/`: Core package containing the RL environment, agent models, draft logic, and utilities.
  - `scripts/`: Entry points for training, simulation, and evaluation.
  - `data/`: Player projections, league matchups, and saved draft states.
  - `models/` & `saved_models/`: Storage for trained policy and value network checkpoints.

## Key Components

### 1. Configuration (`src/draft_buddy/config.py`)
Centralized configuration using dataclasses (`PathsConfig`, `DraftConfig`, `TrainingConfig`, `RewardConfig`, `OpponentConfig`). All major hyperparameters, roster structures, and feature flags are managed here.

### 2. Draft Environment (`src/draft_buddy/draft_env/`)
- `FantasyFootballDraftEnv`: A Gym-compatible environment.
- `DraftState`: Manages the mutable state of the draft (rosters, pick index, history).
- `FantasyDraftRules`: Encapsulates league-specific rules (roster limits, bench capacity).

### 3. Agent & Models (`src/draft_buddy/models/`)
- `ReinforceAgent`: Implements Monte Carlo Policy Gradient with a value-based baseline.
- `PolicyNetwork`: A standard MLP that outputs action probabilities for positions (QB, RB, WR, TE).
- `CheckpointManager`: Handles saving/loading of model weights and optimizer states.

### 4. Logic & Services (`src/draft_buddy/logic/`)
- `OpponentStrategies`: Heuristic, ADP-based, and model-based opponent behaviors.
- `SeasonSimulationService`: Evaluates draft quality by simulating a full season with playoffs.

## Development Workflows

### Environment Setup
The project is Docker-ready. Use `docker-compose up api` to start the web interface.
Alternatively, manually set `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Training
Train the agent using the REINFORCE algorithm:
```bash
python scripts/train.py
```
Progress is logged to `logs/` and checkpoints are saved to `models/`.

### Simulation & Evaluation
Run mock drafts to evaluate a trained model:
```bash
python scripts/simulate.py
```

### Running the Web App
Start the Flask server (defaults to port 5001):
```bash
python api/app.py
```

### Testing
Tests are located in `tests/` and use `pytest`. They use a session-scoped fixture to isolate the file system.
```bash
pytest
```

## Coding Standards & Conventions

- **Docstrings:** Follow NumPy-style docstrings.
- **Linting:** Managed by Ruff (line length 100).
- **Design Principles:** Adhere to SOLID principles, specifically Single Responsibility (SRP) and Dependency Inversion (DIP).
- **Naming:** Use descriptive, searchable names; avoid encodings or prefixes.
- **Testing:** Ensure high coverage for environment logic and reward calculations. One assert per test where possible.
- **Clean Code:** Leave the codebase cleaner than you found it. Eliminate side effects in functions.

- When generating or refactoring code, prioritize understandability, simplicity, and maintainability. In terms of design and architecture, reduce complexity (KISS), use dependency injection, follow the Law of Demeter, and prefer polymorphism over complex conditionals. For naming, use descriptive, searchable, pronounceable names and named constants, keeping names free of encodings, type info, and prefixes. For functions, keep them small and single-purpose, minimize arguments, split methods instead of using flag arguments, and eliminate side effects. For comments, write expressive code, remove commented-out code, and add comments only for complex, non-obvious logic or to warn of consequences. For formatting, use whitespace to separate independent concepts, group related functions together in a downward flow, and declare variables close to their usage. For objects and data, hide internal structures, keep instance variables minimal, and ensure base classes are unaware of their derivatives. For testing, write tests that are fast, independent, repeatable, and readable, and enforce one assert per test. For quality, leave the codebase cleaner than found, eliminate code smells, and always address root causes rather than symptoms.

- Core Architectural Constraints (SOLID) include: Single-Responsibility Principle (SRP): Give every class, module, and function exactly one reason to change. Separate distinct concerns into dedicated classes with descriptive names, and keep each function focused on a single task. Open-Closed Principle (OCP): Design software entities to be open for extension and closed for modification. Add new functionality through subclasses or extended interfaces, using Abstract Base Classes (ABCs) and polymorphism to establish contracts. Liskov Substitution Principle (LSP): Ensure subtypes are fully substitutable for their base types without altering expected behavior. Prefer sibling classes sharing a common abstract interface over parent-child inheritance when it is more logically sound. Interface Segregation Principle (ISP): Define small, role-specific abstract classes so clients depend only on the methods and attributes they use. Dependency Inversion Principle (DIP): Make both high-level and low-level modules depend on abstractions. Inject dependencies through interfaces or abstract base classes (e.g., a generic DataSource rather than a concrete Database class).

- When generating python code, follow these coding standards: All classes, methods, and functions must include NumPy-style docstrings written in clear, concise language. Include comments only where logic is complex or non-obvious. Each function or method should perform a single, focused task. Exclude emojis from all text.

- Strategic replace Tool Usage: The replace tool requires an exact literal match for old_string and is highly sensitive to whitespace. Avoid replacing large, complex blocks of code. Prefer smaller, more targeted replacements. Always re-read the target file immediately before executing a replace command to ensure the old_string is based on the file's current content.

## Data Management
- **Player Data:** Loaded from `data/generated_player_data.csv`.
- **Matchups:** Season simulation depends on matchup CSVs in `data/`.
- **State Persistence:** The web app saves draft state to `data/draft_state.json`.
