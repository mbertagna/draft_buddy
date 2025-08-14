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
        'WR': 2,
        'TE': 1,
        'FLEX': 3, # FLEX can be RB, WR, or TE
    }
    BENCH_MAXES = {
        'QB': 1, # Max QBs on bench
        'RB': 3, # Max RBs on bench
        'WR': 3, # Max WRs on bench
        'TE': 2, # Max TEs on bench
    }
    TOTAL_BENCH_SIZE = 7 # The total number of players allowed on the bench
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
            'logic': 'HEURISTIC',
            'randomness_factor': 0.1,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['RB', 'WR', 'QB', 'TE'] # Custom priority for heuristic logic
        },
        2: { # Example: A more heuristic team that likes QBs early
            'logic': 'HEURISTIC',
            'randomness_factor': 0.2,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['QB', 'RB', 'WR', 'TE']
        },
        3: { # Example: A very random team (might pick odd positions early)
            'logic': 'ADP',
            'randomness_factor': 0.5,
            'suboptimal_strategy': 'NEXT_BEST_ADP',
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
            'randomness_factor': 0.3,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        6: { # Example: A random team with high randomness
            'logic': 'ADP',
            'randomness_factor': 0.7,
            'suboptimal_strategy': 'NEXT_BEST_ADP',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        8: { # Example: An opponent using a trained agent model
            'logic': 'HEURISTIC', # New logic for using a trained model
            'randomness_factor': 0.7,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['WR', 'RB', 'QB', 'TE']
        },
        9: { # Example: Another opponent using a different trained agent model
            'logic': 'HEURISTIC',
            'randomness_factor': 0.1,
            'suboptimal_strategy': 'NEXT_BEST_HEURISTIC',
            'positional_priority': ['RB', 'WR', 'QB', 'TE']
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
        # 'opponent_model_1': os.path.join(MODELS_DIR, "reinforce_policy_model_20250611_194128.pth"),
        # 'opponent_model_2': os.path.join(MODELS_DIR, "reinforce_policy_model_20250612_010014.pth"), # Example
        # 'opponent_model_10': os.path.join(MODELS_DIR, "reinforce_policy_model_20250720_224419.pth"),
        # Add more if you have more agent opponents
    }

    # --- Opponent Strategy Randomization (Optional) ---
    # Enable to randomize non-agent opponent strategies per episode (useful for RL training diversity)
    RANDOMIZE_OPPONENT_STRATEGIES = True
    RANDOMIZE_ONLY_DURING_TRAINING = True  # If True, randomize only when env.training is True
    RANDOMIZE_INCLUDE_AGENT_MODELS = False  # If True, AGENT_MODEL teams can also be randomized (not recommended unless models provided)

    # Strategy templates used for randomization. The env samples one template per team and instantiates parameters.
    # You can tune ranges/choices to fit your league preferences.
    OPPONENT_STRATEGY_TEMPLATES = [
        {
            'logic': 'HEURISTIC',
            'randomness_factor_range': (0.05, 0.45),
            'suboptimal_strategy_choices': ['NEXT_BEST_HEURISTIC', 'NEXT_BEST_ADP'],
            'positional_priority_choices': [
                ['RB', 'WR', 'QB', 'TE'],
                ['WR', 'RB', 'QB', 'TE'],
                ['QB', 'RB', 'WR', 'TE']
            ]
        },
        {
            'logic': 'ADP',
            'randomness_factor_range': (0.1, 0.6),
            'suboptimal_strategy_choices': ['NEXT_BEST_ADP', 'RANDOM_ELIGIBLE'],
            'positional_priority_choices': [
                ['RB', 'WR', 'QB', 'TE'],
                ['WR', 'RB', 'QB', 'TE']
            ]
        },
        {
            'logic': 'RANDOM',
            'randomness_factor_range': (0.0, 1.0),
            'suboptimal_strategy_choices': ['RANDOM_ELIGIBLE'],
            'positional_priority_choices': [
                ['RB', 'WR', 'QB', 'TE']
            ]
        },
    ]

    TEAM_MANAGER_MAPPING = {
        1: 'Ryan Freilich',
        2: 'Shane Spence',
        3: 'Paul Flores',
        4: 'lucas johnsen',
        5: 'Val Perrin',
        6: 'Sean Freilich',
        7: "Jake D'Alonzo",
        8: 'Scott Sheehan',
        9: 'Noah Hollander',
        10: 'Michael Bertagna'
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
    TOTAL_EPISODES = 20000
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.99 # Gamma
    # Variance reduction + exploration
    USE_BASELINE = True
    VALUE_LOSS_COEFFICIENT = 0.5
    ENTROPY_COEFFICIENT = 0.01

    # --- State Space Parameters ---
    ALL_STATE_FEATURES = [
        # --- Original Tier 1 Features ---
        "best_available_qb_points",
        "best_available_rb_points",
        "best_available_wr_points",
        "best_available_te_points",
        "best_available_qb_vorp",
        "best_available_rb_vorp",
        "best_available_wr_vorp",
        "best_available_te_vorp",
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
        "best_qb_bye_week_conflict",
        "best_rb_bye_week_conflict",
        "best_wr_bye_week_conflict",
        "best_te_bye_week_conflict",

        # --- New Tier 2 Features ---

        # 1. Positional Scarcity (Drop-off Score)
        "qb_scarcity",
        "rb_scarcity",
        "wr_scarcity",
        "te_scarcity",

        # 2. Top-k Literal Data (Hybrid Component)
        "top_3_qb_points_1",
        "top_3_qb_points_2",
        "top_3_qb_points_3",
        "top_3_rb_points_1",
        "top_3_rb_points_2",
        "top_3_rb_points_3",
        "top_3_wr_points_1",
        "top_3_wr_points_2",
        "top_3_wr_points_3",
        "top_3_te_points_1",
        "top_3_te_points_2",
        "top_3_te_points_3",

        # 3. Opponent Threat Analysis (Imminent Threat)
        "qb_imminent_threat",
        "rb_imminent_threat",
        "wr_imminent_threat",
        "te_imminent_threat",

        # 4. Bye Week Management (Full Roster Vector)
        "bye_week_4_count",
        "bye_week_5_count",
        "bye_week_6_count",
        "bye_week_7_count",
        "bye_week_8_count",
        "bye_week_9_count",
        "bye_week_10_count",
        "bye_week_11_count",
        "bye_week_12_count",
        "bye_week_13_count",
        "bye_week_14_count",
    ]
    ENABLED_STATE_FEATURES = [
        # --- Original Tier 1 Features ---
        "best_available_qb_points",
        "best_available_rb_points",
        "best_available_wr_points",
        "best_available_te_points",
        "best_available_qb_vorp",
        "best_available_rb_vorp",
        "best_available_wr_vorp",
        "best_available_te_vorp",
        "current_roster_qb_count",
        "current_roster_rb_count",
        "current_roster_wr_count",
        "current_roster_te_count",
        "available_roster_slots_qb",
        "available_roster_slots_rb",
        "available_roster_slots_wr",
        "available_roster_slots_te",
        "available_roster_slots_flex",
        # "qb_available_flag",
        # "rb_available_flag",
        # "wr_available_flag",
        # "te_available_flag",
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
        "best_qb_bye_week_conflict",
        "best_rb_bye_week_conflict",
        "best_wr_bye_week_conflict",
        "best_te_bye_week_conflict",

        # --- New Tier 2 Features ---

        # 1. Positional Scarcity (Drop-off Score)
        "qb_scarcity",
        "rb_scarcity",
        "wr_scarcity",
        "te_scarcity",

        # 2. Top-k Literal Data (Hybrid Component)
        "top_3_qb_points_1",
        "top_3_qb_points_2",
        "top_3_qb_points_3",
        "top_3_rb_points_1",
        "top_3_rb_points_2",
        "top_3_rb_points_3",
        "top_3_wr_points_1",
        "top_3_wr_points_2",
        "top_3_wr_points_3",
        "top_3_te_points_1",
        "top_3_te_points_2",
        "top_3_te_points_3",

        # 3. Opponent Threat Analysis (Imminent Threat)
        "qb_imminent_threat",
        "rb_imminent_threat",
        "wr_imminent_threat",
        "te_imminent_threat",

        # 4. Bye Week Management (Full Roster Vector)
        "bye_week_4_count",
        "bye_week_5_count",
        "bye_week_6_count",
        "bye_week_7_count",
        "bye_week_8_count",
        "bye_week_9_count",
        "bye_week_10_count",
        "bye_week_11_count",
        "bye_week_12_count",
        "bye_week_13_count",
        "bye_week_14_count",
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

    # Per-pick shaping (recommended small signals)
    # 1) Starter-lineup improvement after the pick (delta in starter points)
    ENABLE_PICK_SHAPING_REWARD = False
    PICK_SHAPING_STARTER_DELTA_WEIGHT = 0.25
    # 2) VORP-based shaping to encourage scarcity-aware picks
    ENABLE_VORP_PICK_SHAPING = False
    VORP_PICK_SHAPING_WEIGHT = 0.05

    ENABLE_ROSTER_SLOT_WEIGHTED_REWARD = False
    STARTER_POINTS_WEIGHT = 1
    BENCH_POINTS_WEIGHT = 0.25

    BONUS_FOR_FULL_ROSTER = 0

    # --- Neural Network Architecture ---
    HIDDEN_DIM = 64

    # --- Simulation and Evaluation Parameters ---
    # MODEL_PATH_TO_LOAD = os.path.join(MODELS_DIR, "10_teams_pos_10/v1/checkpoint_episode_30000.pth") # <-- CHANGE THIS FILENAME
    MODEL_PATH_TO_LOAD = os.path.join('saved_models/projected_points/checkpoint_episode_30000.pth') # <-- CHANGE THIS FILENAME
    NUM_SIMULATION_RUNS = 10

    # New: Competitive Reward Parameters
    ENABLE_COMPETITIVE_REWARD = False # Disable competitive difference; we want only playoff win reward
    COMPETITIVE_REWARD_MODE = 'SEASON_SIM' # Keep season sim enabled for playoff win computation

    # --- Season Simulation Reward Parameters ---
    ENABLE_SEASON_SIM_REWARD = True
    # Regular season seeding reward configuration
    # Modes:
    #   - 'LINEAR': interpolate between SEED_REWARD_MAX (seed 1) and SEED_REWARD_MIN (last seed)
    #   - 'MAPPING': use SEED_REWARD_MAPPING dict directly
    REGULAR_SEASON_REWARD = {
        'SEED_REWARD_MODE': 'LINEAR',
        'NUM_PLAYOFF_TEAMS': 6,
        'MAKE_PLAYOFFS_BONUS': 10.0,
        'SEED_REWARD_MAX': 30.0,
        'SEED_REWARD_MIN': 5.0,
        'SEED_REWARD_MAPPING': {
            1: 30.0,
            2: 24.0,
            3: 18.0,
            4: 12.0,
            5: 8.0,
            6: 5.0,
        },
    }
    # Playoff final placement rewards
    PLAYOFF_PLACEMENT_REWARDS = {
        'CHAMPION': 100.0,
        'RUNNER_UP': 40.0,
        'SEMIFINALIST': 15.0,
        'QUARTERFINALIST': 5.0,
        'NON_PLAYOFF': 0.0,
    }

    # Control whether to add the final roster score as a base reward at episode end.
    # Set to False to ensure ONLY season-sim playoff win contributes to final reward.
    ENABLE_FINAL_BASE_REWARD = False

    # Option to add a penalty for high standard deviation among opponents
    ENABLE_OPPONENT_STD_DEV_PENALTY = False # If True, will apply a penalty based on opponent score std dev
    OPPONENT_STD_DEV_PENALTY_WEIGHT = 0.05 # A positive value to penalize high std dev (adjust as needed!)
                                            # Using a smaller value here as a starting point.