import os

class Config:
    """
    Configuration settings for the Fantasy Football Draft AI project.
    """
    # --- Project Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    PLAYER_DATA_CSV = os.path.join(DATA_DIR, 'merged_player_data_cleaned.csv')

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- Draft Environment Parameters ---
    NUM_TEAMS = 10
    AGENT_START_POSITION = 10 # Our agent's pick order (1-indexed)
    ROSTER_STRUCTURE = {
        'QB': 1,
        'RB': 2,
        'WR': 3,
        'TE': 1,
        'FLEX': 2, # FLEX can be RB, WR, or TE
    }
    BENCH_MAXES = {
        'QB': 1, # Max QBs on bench
        'RB': 3, # Max RBs on bench
        'WR': 3, # Max WRs on bench
        'TE': 1, # Max TEs on bench
    }
    COMPETING_TEAM_LOGIC = 'HEURISTIC' # Options: 'ADP', 'HEURISTIC'
    COMPETING_TEAM_RANDOMNESS_FACTOR = 0.2 # Probability (0.0 to 1.0) of making a suboptimal pick
    SUBOPTIMAL_PICK_STRATEGY = 'NEXT_BEST_ADP' # How suboptimal picks are chosen. Options: 'RANDOM_ELIGIBLE', 'NEXT_BEST_ADP', 'NEXT_BEST_HEURISTIC'

    # --- Data Preprocessing Parameters ---
    # Configuration for mock ADP generation if 'adp' column is missing
    MOCK_ADP_CONFIG = {
        'enabled': True, # Set to False if you want an error when adp is missing
        'weights': {
            'projected_points': 1.0, # Higher weight means higher points => lower ADP
            # Add other attributes here if available in CSV and you want to use them
            # 'percentage_on_field': 0.5,
        },
        'sort_order_ascending': False # True for lower values = lower ADP (e.g., age), False for higher values = lower ADP (e.g., points)
    }

    # --- Reinforcement Learning Parameters ---
    TOTAL_EPISODES = 3000
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.99 # Gamma

    # --- State Space Parameters ---
    # A list of all potential state features. The env will use enabled_state_features.
    ALL_STATE_FEATURES = [
        "best_available_qb_points",
        "best_available_rb_points",
        "best_available_wr_points",
        "best_available_te_points",
        "current_roster_qb_count",
        "current_roster_rb_count",
        "current_roster_wr_count",
        "current_roster_te_count",
        "available_roster_slots_qb",
        "available_roster_slots_rb",
        "available_roster_slots_wr",
        "available_roster_slots_te",
        "available_roster_slots_flex",
        "qb_available_flag",
        "rb_available_flag",
        "wr_available_flag",
        "te_available_flag",
        "current_pick_number",
        "agent_start_position",
    ]
    # Select which features to enable for the agent's state vector
    ENABLED_STATE_FEATURES = [
        "best_available_qb_points",
        "best_available_rb_points",
        "best_available_wr_points",
        "best_available_te_points",
        "current_roster_qb_count",
        "current_roster_rb_count",
        "current_roster_wr_count",
        "current_roster_te_count",
        "available_roster_slots_qb",
        "available_roster_slots_rb",
        "available_roster_slots_wr",
        "available_roster_slots_te",
        "available_roster_slots_flex",
        "qb_available_flag",
        "rb_available_flag",
        "wr_available_flag",
        "te_available_flag",
        "current_pick_number",
        "agent_start_position",
    ]
    # State normalization method
    STATE_NORMALIZATION_METHOD = 'min_max' # Options: 'min_max', 'z_score', 'none'

    # --- Action Masking Parameter ---
    ENABLE_ACTION_MASKING = True # Set to True to enable action masking

    # --- Reward Function Parameters ---
    INVALID_ACTION_PENALTIES = {
    'already_drafted': -100, # Much smaller penalty
    'roster_full_QB': -50,
    'roster_full_RB': -40,
    'roster_full_WR': -40,
    'roster_full_TE': -50,
    'no_players_available': -100, # Added this penalty key previously for completeness
    'default_invalid': -50
    }
    # Flag to enable/disable invalid action penalties
    ENABLE_INVALID_ACTION_PENALTIES = True # Set to False for "simple reward mode"

    ENABLE_INTERMEDIATE_REWARD = True # True for small reward for valid picks
    INTERMEDIATE_REWARD_MODE = 'PROPORTIONAL' # New: 'STATIC' or 'PROPORTIONAL'
    INTERMEDIATE_REWARD_VALUE = 30 # Static reward for each successful valid pick if enabled and mode is 'STATIC'
    PROPORTIONAL_REWARD_SCALING_FACTOR = 1 # New: Multiplier for projected points when mode is 'PROPORTIONAL'


    # Bonus for successfully drafting an entire team
    BONUS_FOR_FULL_ROSTER = 0 # A significant positive bonus (e.g., 100 or 200)

    # --- Neural Network Architecture ---
    HIDDEN_DIM = 64

    # --- Simulation and Evaluation Parameters ---
    # Path to a trained model (.pth file) to load for simulation/evaluation
    MODEL_PATH_TO_LOAD = os.path.join(MODELS_DIR, "reinforce_policy_model_20250611_034822.pth") # REPLACE with your actual model path
    NUM_SIMULATION_RUNS = 5 # How many times to run the draft simulation

    # You might want to temporarily override some env settings for simulation if different from training
    # SIM_COMPETING_TEAM_LOGIC = 'ADP' # For example, always test against ADP opponents
    # SIM_AGENT_START_POSITION = 7 # Test a specific draft position

# You can access configuration like: `Config.TOTAL_EPISODES`