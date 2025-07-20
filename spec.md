## DRAFT BUDDY: Web-Based Fantasy Football Draft Dashboard - Technical Specification

### 1. Introduction

"DRAFT BUDDY" is a web-based dashboard designed to assist fantasy football managers during live or mock drafts. It provides a streamlined interface for a single user to track draft progress, manage their roster, observe opponent team compositions, and utilize AI insights. The application is built upon an existing Python codebase focused on reinforcement learning for draft simulation.

### 2. Core Functional Requirements

#### 2.1. Frontend Layout and Features

The user interface will be organized into the following key sections:

* **Main Player List:**
    * Displays all available players.
    * Includes a text search bar for player names.
    * Provides multi-select checkboxes for filtering players by position (QB, RB, WR, TE).
    * Predetermined data columns for each player: `player_id`, `name`, `position`, `projected_points`, `adp`.
* **Team Rosters View:**
    * Dedicated area to display current players for all 10 teams in the draft.
    * Provides an at-a-glance overview of team compositions.
* **Draft Summary View:**
    * Panel offering high-level statistics about the draft's progress.
    * Examples: Total number of players drafted for each position (QB, RB, WR, TE).
* **Draft Control Panel:**
    * Central hub displaying the current pick number and the team "on the clock."
    * 10 clickable buttons, one for each team, with the current team's button highlighted.
    * "Undo Last Pick" button to revert the most recent draft selection.
    * Toggle switch for "Manual Step" or "Auto-Play" modes to control simulation pace.
    * "Download as CSV" button to export draft results.
* **AI Agent as Advisor Display:**
    * When a manual team is "on the clock," the system will display the AI agent's suggested position (e.g., "Agent suggests: WR"). The agent will *not* automatically make the pick.

#### 2.2. Key Frontend Interactions

* **Draft Setup:**
    * **Draft Order:** Before the draft begins, managers can manually edit the snake draft order to accommodate pick trades.
    * **Drafting Method per Team:** Each of the 10 teams can be assigned a drafting method:
        * Manual (user controls picks for this team)
        * ADP (computer drafts based on Average Draft Position)
        * Heuristic (computer drafts based on pre-defined positional priorities and roster needs)
        * Agent Model (computer drafts using a specific pre-trained AI model from the `models/` directory, configurable in `config.py`)
* **Manual Drafting:**
    * To make a pick, the manager clicks on an available player from the "Main Player List."
    * The player is immediately assigned to the currently highlighted "on-the-clock" team.
    * No confirmation pop-up is displayed to ensure quick drafting.
    * Managers can override the current "on-the-clock" team by clicking any of the 10 team buttons in the "Draft Control Panel."
* **Automated Drafting:**
    * **"Manual Step" Mode:** Requires the manager to click a "Simulate Pick" button for each computer-controlled selection.
    * **"Auto-Play" Mode:** The system automatically makes simulated picks with a 2-second delay. It will pause only when a manual team is "on the clock."
* **Undo Functionality:**
    * The "Undo Last Pick" button reverts the most recent draft selection.
    * The player is returned to the "Main Player List" (available pool).
    * The draft state (current pick number, team on the clock, team rosters) is reset to the state before that pick.

#### 2.3. Backend Logic & Integration (Leveraging Existing Codebase)

The existing Python codebase provides a strong foundation for the backend logic and simulation:

* **Player Data Loading (`data_utils.py`):**
    * The `load_player_data` function will be used to load player information from `data/merged_player_data_cleaned.csv`.
    * It supports mock ADP generation if the ADP column is missing or empty.
* **Draft Environment Simulation (`fantasy_draft_env.py`):**
    * The `FantasyFootballDraftEnv` class provides the core draft logic.
    * It manages available players, team rosters, and simulates picks by competing teams.
    * The `_generate_snake_draft_order` function will be used as a base for generating draft orders. Manual editing of this order will require an additional frontend-driven update to this mechanism.
    * `_simulate_competing_pick` will be extended/adapted to dynamically select opponent strategies (ADP, Heuristic, Agent Model) based on the user's `config.py` setup.
    * `_get_state` and `_get_info` can be used to feed data to the frontend for display.
    * `get_action_mask` logic is crucial for guiding the AI and can also inform UI elements.
    * `_can_team_draft_position` and `_try_select_player_for_team` are fundamental for validating picks, whether manual or automated.
* **AI Agent Integration (`policy_network.py`, `reinforce_agent.py`, `simulate.py`, `config.py`):**
    * Pre-trained `PolicyNetwork` models (saved as `.pth` files in `models/`) will be loaded and utilized for "Agent Model" drafting methods (for both opponent teams and the user's AI advisor).
    * The `simulate.py` module demonstrates how to load and use trained models for making picks. This logic will be adapted for real-time pick generation within the web application.
    * The `Config` class (`config.py`) will be the central point for pre-configuring:
        * `NUM_TEAMS`, `ROSTER_STRUCTURE`, `BENCH_MAXES`
        * `OPPONENT_TEAM_STRATEGIES` and `OPPONENT_MODEL_PATHS` for defining opponent personalities and their associated trained models.
        * `AGENT_START_POSITION` will define which team the user controls.

#### 2.4. Data Persistence & Export

* **Configuration:** All draft setup parameters (league size, roster structure, opponent AI types, and paths to AI models) will be loaded from `config.py`. Changes to these settings will require direct modification of `config.py` and a restart of the application (no in-UI saving of these settings).
* **Download as CSV:** The "Download as CSV" button will export *all* draft data from the current session, including every pick made by every team, along with player details (`player_id`, `name`, `position`, `projected_points`, `adp`).

### 3. Technical Architecture (High-Level)

* **Frontend:** HTML, CSS, JavaScript (e.g., a modern JS framework like React, Vue, or Angular, or even vanilla JS for simplicity).
* **Backend:** Python (Flask, FastAPI, or Django could be options to serve the web application and interact with the core draft logic).
* **Communication:** REST API or WebSockets for real-time updates between frontend and backend during the draft.
* **Deployment:** Docker (as indicated by `Dockerfile`, `build.sh`, `run.sh`) for containerization, ensuring consistent environments for development and deployment.

### 4. Implementation Details & Considerations

* **Real-time Updates:** For a smooth user experience, the frontend will need real-time updates as picks are made (especially in "Auto-Play" mode or when opponent AI makes a selection). WebSockets would be ideal for this.
* **UI Responsiveness:** The UI should remain responsive during automated picks, perhaps showing a progress indicator or a visual delay.
* **Error Handling:** Robust error handling will be needed, especially for invalid user actions or issues with loading AI models or player data.
* **Scalability:** While initially for a single user, consider the architecture for potential future multi-user support (though not in the current spec).
* **AI Advisor Integration:** The "Agent suggests: WR" feature will involve running the AI model on the backend for the current state and sending the suggested pick back to the frontend for display.

### 5. Future Enhancements (Out of Scope for Initial Spec but good to note)

* User accounts and saved league profiles.
* More sophisticated opponent AI personalities (e.g., dynamically changing strategies).
* Integration with live draft data APIs (ESPN, Yahoo, etc.).
* Customizable player data fields.
* Advanced draft analytics and visualizations.
