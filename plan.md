## Implementation Plan: DRAFT BUDDY

Here's a proposed plan, broken down into logical stages. Each stage builds upon the previous one, minimizing dependencies and allowing for iterative development.

### Stage 1: Backend Core Setup & Data Handling

This stage focuses on getting the foundational data and draft logic accessible and testable.

* **1.1 Project Initialization & Environment:**
    * Set up the basic project structure for the web application (e.g., create a new directory for the frontend, define initial API endpoints).
    * Ensure the Docker setup is ready for running both the backend and potentially serving the frontend.
* **1.2 Player Data API:**
    * Create a backend API endpoint that loads and exposes the player data (`merged_player_data_cleaned.csv`) via HTTP. This endpoint should be able to filter by position and potentially support searching.
    * This will leverage your existing `data_utils.py` for data loading.
* **1.3 Initial Draft Environment State API:**
    * Create a backend API endpoint that initializes a new `FantasyFootballDraftEnv` instance.
    * This endpoint should return the initial state of the draft environment, including the generated draft order, current pick number, and initial roster counts for all teams.
    * It will leverage the `reset()` method of `FantasyFootballDraftEnv`.

### Stage 2: Frontend Core Layout & Static Data Display

This stage focuses on building the basic UI and displaying the static (non-interactive) data from the backend.

* **2.1 Basic Web Server & Frontend Framework Setup:**
    * Choose a lightweight frontend framework (or vanilla JS) and set up the basic web server to serve HTML, CSS, and JavaScript files.
* **2.2 Main Player List Display:**
    * Build the "Main Player List" section of the UI.
    * Consume the Player Data API (from Stage 1.2) to populate the list with players, their positions, projected points, and ADP.
* **2.3 Team Rosters & Draft Summary Display:**
    * Build the "Team Rosters View" and "Draft Summary View" sections.
    * Consume the Initial Draft Environment State API (from Stage 1.3) to display initial team roster information (e.g., "QB: 0/1, RB: 0/2") and high-level draft statistics.

### Stage 3: Draft Control & Manual Interaction

This stage introduces the core user interaction for making picks.

* **3.1 Draft Control Panel UI:**
    * Implement the "Draft Control Panel" UI elements: current pick number display, "on-the-clock" team indicator, and the 10 clickable team buttons.
* **3.2 Make Pick API Endpoint:**
    * Create a backend API endpoint that accepts a `team_id` and a `player_id` (or `position_choice`).
    * This endpoint will interact with `FantasyFootballDraftEnv`'s internal logic to "make a pick," update the environment, and return the new state. This will be an adapted version of your `step()` method.
* **3.3 Manual Player Selection (UI to API):**
    * Connect the "Main Player List" to the "Make Pick API Endpoint." When the user clicks on a player, trigger the API call.
    * Update the frontend (Main Player List, Team Rosters, Draft Summary) to reflect the changes after a successful pick.

### Stage 4: Automated Drafting & Undo Functionality

This stage adds the automated simulation and the ability to revert picks.

* **4.1 Simulate Pick API Endpoint:**
    * Create a backend API endpoint that triggers `_simulate_competing_pick` for the current "on-the-clock" team and advances the draft state.
* **4.2 Auto-Play/Manual Step Controls:**
    * Implement the "Manual Step" button (which calls the Simulate Pick API once).
    * Implement the "Auto-Play" toggle, which repeatedly calls the Simulate Pick API with a delay until a manual team is up.
* **4.3 Undo Last Pick API & UI:**
    * Implement backend logic and an API endpoint for "Undo Last Pick." This will require storing a history of draft states on the backend.
    * Connect the "Undo Last Pick" button to this API endpoint and update the frontend accordingly.

### Stage 5: AI Advisor & Enhanced Information Display

This stage integrates the AI advice and refines the information presented to the user.

* **5.1 AI Advisor API:**
    * Create a backend API endpoint that, when called for the manual team's turn, uses your loaded `PolicyNetwork` to suggest an optimal position. This will adapt the agent's logic from `simulate.py` but return only the *suggested position*.
* **5.2 AI Advisor Display in UI:**
    * Display the AI agent's suggested position (e.g., "Agent suggests: WR") in the "Draft Control Panel" when the manual team is "on the clock."
* **5.3 Refined Draft Summary & Roster Display:**
    * Enhance the "Draft Summary View" with more detailed statistics (e.g., positional needs across all teams, average projected points per roster slot).
    * Improve the "Team Rosters View" to clearly show starters, bench, and flex players for each team, potentially with their individual projected points.

### Stage 6: Export & Final Polish

This final stage focuses on the export functionality and overall user experience.

* **6.1 Download as CSV API:**
    * Create a backend API endpoint that gathers all draft log data and formats it into a CSV.
* **6.2 Download as CSV UI Integration:**
    * Connect the "Download as CSV" button to this API endpoint, allowing users to download the full draft log.
* **6.3 UI/UX Refinements:**
    * Focus on visual polish, responsiveness, and overall user experience.
    * Add loading indicators, error messages, and intuitive navigation.
