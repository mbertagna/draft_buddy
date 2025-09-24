# Draft Buddy 🏈 - An AI-Powered Fantasy Football Draft Assistant

Draft Buddy is a complete system for simulating, training, and running a fantasy football draft assistant. It leverages reinforcement learning to train an AI agent to draft an optimal team by understanding player value, positional scarcity, and opponent behavior.

This project is more than just a draft simulator; it's a powerful tool for aspiring GMs to test strategies, train a personalized AI advisor, and get real-time suggestions during a live draft.

-----

## ✨ Key Features

  * **Customizable Draft Environment**: A custom OpenAI Gym environment models a multi-team snake draft with realistic roster rules, including a FLEX position.
  * **Intelligent Opponent Strategies**: Pit your AI against a range of opponent "personalities" from rule-based strategies (`ADP`, `HEURISTIC`, `RANDOM`) to other trained AI models. These can be randomized on the fly for robust training.
  * **Rich State Representation**: The agent's decision-making is powered by a comprehensive observation space that includes player value (`VORP`), positional scarcity, and opponent roster composition.
  * **Advanced Reward Functions**: Train your agent with flexible reward configurations, including per-pick shaping and a unique end-of-episode reward based on a full-season simulation. This rewards the agent for building a team that actually wins games, not just one with the highest projected points.
  * **Interactive Web UI**: A simple Flask backend and lightweight frontend allow you to run mock drafts, manually make picks, get real-time AI suggestions, and view detailed roster breakdowns.
  * **Extensive Analytics**: Simulate entire seasons to evaluate team performance, analyze draft results with CSV exports, and visualize training progress with intuitive plots.
  * **Docker-Ready**: The entire system is containerized, ensuring a consistent and isolated environment for all dependencies and making it easy to run anywhere.

-----

## 🚀 Getting Started

The easiest way to get started is by using Docker.

### Prerequisites

  * [Docker](https://www.docker.com/get-started) installed on your system.

### Installation & Setup

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/draft-buddy.git
    cd draft-buddy
    ```

2.  **Build the Docker image**:
    This command builds the Docker image and installs all necessary Python and system dependencies.

    ```bash
    ./build.sh
    ```

3.  **Run the container**:
    This command starts a container, mounts your local project directory inside, and drops you into a shell. Any changes you make locally will be immediately reflected in the container.

    ```bash
    ./run.sh
    ```

-----

## 🕹️ Running the Web App (UI)

Once inside the Docker container, you can start the Flask web server.

1.  **Start the server**:

    ```bash
    python app.py
    ```

2.  **Open the web UI**:
    Open your browser and navigate to `http://localhost:8000`.

The UI provides real-time controls and visualizations:

  * `Start New Draft`: Clears the current state and begins a new draft.
  * `Sim Pick`: Has the current team on the clock make an automatic selection.
  * `Undo`: Reverts the last pick.
  * `CSV`: Exports the entire draft history to a CSV file.
  * **Player List**: Filter, search, and sort through the available player pool.
  * **Team Rosters**: See a live breakdown of each team's roster, including starters, bench, and bye week conflicts.
  * **AI Suggestions**: Get real-time AI recommendations for the team on the clock.
  * **Sim Season**: Run a full season simulation based on current rosters to test the effectiveness of your draft.

-----

## 📈 Training the AI Agent

The core of Draft Buddy is the reinforcement learning agent trained with the REINFORCE algorithm.

1.  **Prepare your configuration**:
    Open `config.py` and adjust parameters such as `TOTAL_EPISODES`, `LEARNING_RATE`, and `ENABLED_STATE_FEATURES`. Pay special attention to the `ENABLE_SEASON_SIM_REWARD` flag if you want to train the agent to win a simulated season.

2.  **Start training**:
    From the Docker shell, run the `train.py` script.

    ```bash
    python train.py
    ```

    Training progress, including rewards and losses, will be logged to the `logs/` directory.

3.  **Resume training**:
    Set `RESUME_TRAINING = True` in `config.py` and the script will automatically find and load the latest checkpoint to continue training.

4.  **Plotting results**:
    To visualize your training metrics without starting a new training run, use the `-p` flag.

    ```bash
    python train.py -p
    ```

    This generates an interactive HTML dashboard in the `logs/` directory.

-----

## 🧪 Simulation & Evaluation

Use a trained model to run multiple mock drafts and evaluate its performance.

1.  **Update the model path**:
    In `config.py`, ensure `MODEL_PATH_TO_LOAD` points to the `.pth` file of the trained agent you want to evaluate.

2.  **Run the simulation**:

    ```bash
    python simulate.py
    ```

    The script will output a detailed log of each pick and a summary of final team scores across all simulation runs, allowing you to see how your agent stacks up against its opponents.

-----

## 🛠️ Project Structure

```
.
├── app.py                      # Flask API and web UI backend
├── config.py                   # Central configuration for all components
├── data/                       # Stores player data, draft states, and matchup files
├── data_driver.py              # Script to process raw data and generate a player pool
├── fantasy_draft_env.py        # The core OpenAI Gym environment
├── policy_network.py           # The neural network architecture for the agent
├── reinforce_agent.py          # The REINFORCE training algorithm implementation
├── requirements.txt            # Python dependencies
├── run.sh, build.sh            # Scripts for managing the Docker environment
├── simulate.py                 # Script to evaluate a trained model on mock drafts
├── train.py                    # Script to train the reinforcement learning agent
└── utils/                      # Helper scripts for data processing, simulation, etc.
```

-----

## 📄 License & Acknowledgments

This project is open-sourced under the **MIT License**.

A special thanks to the open-source community behind Python, Gym, PyTorch, Pandas, Flask, and the various data sources used in this project. All player projections and logic should be adapted to your specific league's rules and data sources.