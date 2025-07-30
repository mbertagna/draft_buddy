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
    PLAYER_DATA_CSV = os.path.join(DATA_DIR, 'generated_player_data.csv')
    DRAFT_STATE_FILE = os.path.join(DATA_DIR, 'draft_state.json')

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- Draft Environment Parameters ---
    NUM_TEAMS = 10
    AGENT_START_POSITION = 10 # Our agent's pick order (1-indexed) - used for RL training
    MANUAL_DRAFT_TEAMS = [
                        # 1, 
                        # 2, 
                        # 3, 
                        # 4, 
                        # 5, 
                        # 6, 
                        # 7, 
                        # 8, 
                        # 9, 
                        10,
                        ] # List of team IDs (1-indexed) that will be controlled manually by the user
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
            'randomness_factor': 0.2,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['QB', 'RB', 'WR', 'TE']
        },
        3: { # Example: A very random team (might pick odd positions early)
            'logic': 'RANDOM',
            'randomness_factor': 0.5,
            'suboptimal_strategy': 'RANDOM_ELIGIBLE',
            'positional_priority': ['RB', 'WR', 'QB', 'TE']
        },
        4: { # Example: Another ADP team with higher randomness
            'logic': 'ADP',
            'randomness_factor': 0.3,
            'suboptimal_strategy': 'NEXT_BEST_ADP',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        5: { # Example: A heuristic team prioritizing WR/RB
            'logic': 'HEURISTIC',
            'randomness_factor': 0.1,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        6: { # Example: A random team with high randomness
            'logic': 'RANDOM',
            'randomness_factor': 0.7,
            'suboptimal_strategy': 'RANDOM_ELIGIBLE',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        7: { # Example: An opponent using a trained agent model
            'logic': 'AGENT_MODEL', # New logic for using a trained model
            'model_path_key': 'opponent_model_1', # Key to look up in OPPONENT_MODEL_PATHS
            # randomness_factor, suboptimal_strategy, positional_priority irrelevant here
        },
        8: { # Example: Another opponent using a different trained agent model
            'logic': 'AGENT_MODEL',
            'model_path_key': 'opponent_model_2',
        },
        # 10: { # Example: Another opponent using a different trained agent model
        #     'logic': 'AGENT_MODEL',
        #     'model_path_key': 'opponent_model_10',
        # },
        # You can add more AGENT_MODEL opponents as needed
    }

    # Default strategy for any opponent team not explicitly defined in OPPONENT_TEAM_STRATEGIES
    DEFAULT_OPPONENT_STRATEGY = {
        'logic': 'HEURISTIC',
        'randomness_factor': 0.2,
        'suboptimal_strategy': 'NEXT_BEST_ADP',
        'positional_priority': ['RB', 'WR', 'QB', 'TE']
    }

    # New: Paths to trained opponent models
    # Each key should match a 'model_path_key' in OPPONENT_TEAM_STRATEGIES
    OPPONENT_MODEL_PATHS = {
        # IMPORTANT: Replace these with actual paths to your trained .pth files
        'opponent_model_1': os.path.join(MODELS_DIR, "reinforce_policy_model_20250611_194128.pth"),
        'opponent_model_2': os.path.join(MODELS_DIR, "reinforce_policy_model_20250612_010014.pth"), # Example
        # 'opponent_model_10': os.path.join(MODELS_DIR, "reinforce_policy_model_20250720_224419.pth"),
        # Add more if you have more agent opponents
    }


    # --- Data Preprocessing Parameters ---
    # Configuration for mock ADP generation if 'adp' column is missing
    MOCK_ADP_CONFIG = {
        'enabled': True,
        'weights': {
            'projected_points': 1.0,
        },
        'sort_order_ascending': False
    }

    # --- Reinforcement Learning Parameters ---
    RESUME_TRAINING = True # Set to True to resume from the latest checkpoint
    TOTAL_EPISODES = 2000
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.99 # Gamma

    # --- State Space Parameters ---
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
        "second_best_available_qb_points",
        "second_best_available_rb_points",
        "second_best_available_wr_points",
        "second_best_available_te_points",
        "next_pick_opponent_qb_count",
        "next_pick_opponent_rb_count",
        "next_pick_opponent_wr_count",
        "next_pick_opponent_te_count",
    ]
    ENABLED_STATE_FEATURES = [
        "best_available_qb_points",
        "best_available_rb_points",
        "best_available_wr_points",
        "best_available_te_points", # Keep TE in here to make sure this is complete
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
        "second_best_available_qb_points", # Adding this back for completeness in state features
        "second_best_available_rb_points",
        "second_best_available_wr_points",
        "second_best_available_te_points", # Adding this back for completeness in state features
        "next_pick_opponent_qb_count", # Adding this back for completeness in state features
        "next_pick_opponent_rb_count",
        "next_pick_opponent_wr_count",
        "next_pick_opponent_te_count", # Adding this back for completeness in state features
    ]
    STATE_NORMALIZATION_METHOD = 'min_max'

    # --- Action Masking Parameter ---
    ENABLE_ACTION_MASKING = True

    # --- Reward Function Parameters ---
    INVALID_ACTION_PENALTIES = {
    'already_drafted': -100,
    'roster_full_QB': -50,
    'roster_full_RB': -40,
    'roster_full_WR': -40,
    'roster_full_TE': -50,
    'no_players_available': -100,
    'default_invalid': -50
    }
    ENABLE_INVALID_ACTION_PENALTIES = False

    ENABLE_INTERMEDIATE_REWARD = False
    INTERMEDIATE_REWARD_MODE = 'PROPORTIONAL'
    INTERMEDIATE_REWARD_VALUE = 30
    PROPORTIONAL_REWARD_SCALING_FACTOR = 1

    ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = True
    STARTER_POINTS_WEIGHT = 1
    BENCH_POINTS_WEIGHT = 0.25

    BONUS_FOR_FULL_ROSTER = 0

    # --- Neural Network Architecture ---
    HIDDEN_DIM = 64

    # --- Simulation and Evaluation Parameters ---
    MODEL_PATH_TO_LOAD = os.path.join(MODELS_DIR, "10_teams_pos_10/v1/checkpoint_episode_2000.pth") # <-- CHANGE THIS FILENAME
    NUM_SIMULATION_RUNS = 10

    # New: Competitive Reward Parameters
    ENABLE_COMPETITIVE_REWARD = False # Master switch for competitive rewards
    COMPETITIVE_REWARD_MODE = 'MAX_OPPONENT_DIFFERENCE' # Options: 'NONE', 'MAX_OPPONENT_DIFFERENCE', 'AVG_OPPONENT_DIFFERENCE'

    # Option to add a penalty for high standard deviation among opponents
    ENABLE_OPPONENT_STD_DEV_PENALTY = False # If True, will apply a penalty based on opponent score std dev
    OPPONENT_STD_DEV_PENALTY_WEIGHT = 0.05 # A positive value to penalize high std dev (adjust as needed!)
                                            # Using a smaller value here as a starting point.