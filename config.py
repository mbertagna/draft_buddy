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
    # These base values are now used only for agent's own settings, or as default if not overridden for opponents
    COMPETING_TEAM_LOGIC = 'HEURISTIC' # Options: 'ADP', 'HEURISTIC'
    COMPETING_TEAM_RANDOMNESS_FACTOR = 0.2 # Probability (0.0 to 1.0) of making a suboptimal pick
    SUBOPTIMAL_PICK_STRATEGY = 'NEXT_BEST_ADP' # How suboptimal picks are chosen. Options: 'RANDOM_ELIGIBLE', 'NEXT_BEST_ADP', 'NEXT_BEST_HEURISTIC'

    # New: Opponent Personalities
    # Define strategies for specific opponent teams.
    # Team IDs are 1-indexed. Agent's team ID (AGENT_START_POSITION) should NOT be here.
    # If a team_id is not specified here, it will use DEFAULT_OPPONENT_STRATEGY.
    OPPONENT_TEAM_STRATEGIES = {
        1: { # Example: A team that prioritizes ADP with some randomness, and focuses on RB/WR first
            'logic': 'ADP',
            'randomness_factor': 0.1,
            'suboptimal_strategy': 'NEXT_BEST_ADP',
            'positional_priority': ['RB', 'WR', 'QB', 'TE'] # Custom priority for heuristic logic
        },
        2: { # Example: A more heuristic team that likes QBs early
            'logic': 'HEURISTIC',
            'randomness_factor': 0.0, # Very deterministic
            'suboptimal_strategy': 'NONE', # N/A if randomness is 0
            'positional_priority': ['QB', 'RB', 'WR', 'TE']
        },
        3: { # Example: A very random team (might pick odd positions early)
            'logic': 'RANDOM', # New 'RANDOM' logic that picks any eligible player
            'randomness_factor': 1.0,
            'suboptimal_strategy': 'RANDOM_ELIGIBLE',
            'positional_priority': ['RB', 'WR', 'QB', 'TE'] # Irrelevant for 'RANDOM' logic but kept for consistency
        },
        4: { # Example: A team that prioritizes ADP with some randomness, and focuses on RB/WR first
            'logic': 'ADP',
            'randomness_factor': 0.1,
            'suboptimal_strategy': 'NEXT_BEST_ADP',
            'positional_priority': ['WR', 'RB', 'QB', 'TE'] # Custom priority for heuristic logic
        },
        5: { # Example: A more heuristic team that likes QBs early
            'logic': 'HEURISTIC',
            'randomness_factor': 0.0, # Very deterministic
            'suboptimal_strategy': 'NONE', # N/A if randomness is 0
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        6: { # Example: A very random team (might pick odd positions early)
            'logic': 'RANDOM', # New 'RANDOM' logic that picks any eligible player
            'randomness_factor': 1.0,
            'suboptimal_strategy': 'RANDOM_ELIGIBLE',
            'positional_priority': ['WR', 'RB', 'QB', 'TE'] # Irrelevant for 'RANDOM' logic but kept for consistency
        },
        # Add more teams here, e.g., teams 4-9
        # 4: {'logic': 'HEURISTIC', 'randomness_factor': 0.3, 'suboptimal_strategy': 'RANDOM_ELIGIBLE', 'positional_priority': ['WR', 'RB', 'TE', 'QB']},
        # etc.
    }

    # Default strategy for any opponent team not explicitly defined in OPPONENT_TEAM_STRATEGIES
    DEFAULT_OPPONENT_STRATEGY = {
        'logic': 'HEURISTIC',
        'randomness_factor': 0.2,
        'suboptimal_strategy': 'NEXT_BEST_ADP',
        'positional_priority': ['RB', 'WR', 'QB', 'TE'] # Default human-like priority
    }


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
        # New features below
        "second_best_available_qb_points", # New
        "second_best_available_rb_points", # New
        "second_best_available_wr_points", # New
        "second_best_available_te_points", # New
        "next_pick_opponent_qb_count", # New
        "next_pick_opponent_rb_count", # New
        "next_pick_opponent_wr_count", # New
        "next_pick_opponent_te_count", # New
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
        "second_best_available_rb_points", # Let's enable this one for RB cliff awareness
        "second_best_available_wr_points", # Let's enable this one for WR cliff awareness
        "next_pick_opponent_rb_count", # Critical for RB denial
        "next_pick_opponent_wr_count", # Critical for WR denial
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
    ENABLE_INVALID_ACTION_PENALTIES = False # Set to False for "simple reward mode"

    ENABLE_INTERMEDIATE_REWARD = False # True for small reward for valid picks
    INTERMEDIATE_REWARD_MODE = 'PROPORTIONAL' # 'STATIC' or 'PROPORTIONAL'
    INTERMEDIATE_REWARD_VALUE = 30 # Static reward for each successful valid pick if enabled and mode is 'STATIC'
    PROPORTIONAL_REWARD_SCALING_FACTOR = 1 # Multiplier for projected points when mode is 'PROPORTIONAL'


    # Bonus for successfully drafting an entire team
    BONUS_FOR_FULL_ROSTER = 0 # A significant positive bonus (e.g., 100 or 200)

    # --- Neural Network Architecture ---
    HIDDEN_DIM = 64

    # --- Simulation and Evaluation Parameters ---
    # Path to a trained model (.pth file) to load for simulation/evaluation
    MODEL_PATH_TO_LOAD = os.path.join(MODELS_DIR, "reinforce_policy_model_20250611_073210.pth") # REPLACE with your actual model path
    NUM_SIMULATION_RUNS = 5 # How many times to run the draft simulation

    # You might want to temporarily override some env settings for simulation if different from training
    # SIM_COMPETING_TEAM_LOGIC = 'ADP' # For example, always test against ADP opponents
    # SIM_AGENT_START_POSITION = 7 # Test a specific draft position

# You can access configuration like: `Config.TOTAL_EPISODES`