"""Runtime configuration settings for Draft Buddy."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_INT_KEYED_DRAFT_FIELDS = frozenset({"TEAM_MANAGER_MAPPING"})
_INT_KEYED_OPPONENT_FIELDS = frozenset({"OPPONENT_TEAM_STRATEGIES"})


def repository_root() -> str:
    """Return the repository root directory.

    Returns
    -------
    str
        Absolute path to the project root (parent of ``src/``).
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class LeagueMetaConfig:
    """Stable league identity metadata."""

    league_id: str = ""
    display_name: str = ""


@dataclass
class ScoringConfig:
    """Fantasy scoring rules keyed by stat name."""

    rules: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class SeasonRuntimeConfig:
    """Season-specific runtime values loaded from season overlays."""

    season: int = 2026
    bye_weeks: Dict[int, List[str]] = field(default_factory=dict)
    position_guide_checkpoint_dir: str = ""


@dataclass
class PathsConfig:
    """Configuration for project directory and file paths."""

    BASE_DIR: str = field(default_factory=repository_root)
    DATA_DIR: str = field(init=False)
    MODELS_DIR: str = field(init=False)
    LOGS_DIR: str = field(init=False)
    PLAYER_DATA_CSV: str = field(init=False)
    PLAYER_DATA_TEMPLATE: str = ""
    DRAFT_STATE_FILE: str = field(init=False)
    DRAFT_STATE_PREV_FILE: str = field(init=False)
    SAVED_STATES_DIR: str = field(init=False)

    def __post_init__(self) -> None:
        """Derive standard paths from the repository base directory."""
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models")
        self.LOGS_DIR = os.path.join(self.BASE_DIR, "logs")
        if not self.PLAYER_DATA_TEMPLATE:
            self.PLAYER_DATA_CSV = os.path.join(self.DATA_DIR, "generated_player_data.csv")
        self.DRAFT_STATE_FILE = os.path.join(self.DATA_DIR, "draft_state.json")
        self.DRAFT_STATE_PREV_FILE = os.path.join(self.DATA_DIR, "draft_state.prev.json")
        self.SAVED_STATES_DIR = os.path.join(self.BASE_DIR, "saved_states")

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.SAVED_STATES_DIR, exist_ok=True)


@dataclass
class DraftConfig:
    """Configuration for the draft environment and league rules."""

    NUM_TEAMS: int = 12
    ACTION_SPACE_SIZE: int = 4
    AGENT_START_POSITION: int = 5
    RANDOMIZE_AGENT_START_POSITION: bool = True
    MANUAL_DRAFT_TEAMS: List[int] = field(default_factory=list)
    ROSTER_STRUCTURE: Dict[str, int] = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
    )
    BENCH_MAXES: Dict[str, int] = field(
        default_factory=lambda: {"QB": 1, "RB": 3, "WR": 3, "TE": 2}
    )
    TOTAL_BENCH_SIZE: int = 6
    # Cosmetic display names only (UI / LLM prompts). Never use as compute keys.
    TEAM_MANAGER_MAPPING: Dict[int, str] = field(
        default_factory=lambda: {
            1: "Ryan Freilich",
            2: "Shane Spence",
            3: "Paul Flores",
            4: "lucas johnsen",
            5: "Val Perrin",
            6: "Sean Freilich",
            7: "Jake D'Alonzo",
            8: "Scott Sheehan",
            9: "Noah Hollander",
            10: "Michael Bertagna",
            11: "Manager 11",
            12: "Manager 12",
        }
    )
    MOCK_ADP_CONFIG: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "weights": {"projected_points": 1.0},
            "sort_order_ascending": False,
        }
    )
    TEAM_ABBREVIATIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "Detroit Lions": "DET",
            "Los Angeles Chargers": "LAC",
            "Philadelphia Eagles": "PHI",
            "Tennessee Titans": "TEN",
            "Kansas City Chiefs": "KC",
            "Los Angeles Rams": "LA",
            "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN",
            "Chicago Bears": "CHI",
            "Dallas Cowboys": "DAL",
            "Pittsburgh Steelers": "PIT",
            "San Francisco 49ers": "SF",
            "Cleveland Browns": "CLE",
            "Green Bay Packers": "GB",
            "Las Vegas Raiders": "LV",
            "Seattle Seahawks": "SEA",
            "Arizona Cardinals": "ARI",
            "Carolina Panthers": "CAR",
            "New York Giants": "NYG",
            "Tampa Bay Buccaneers": "TB",
            "Atlanta Falcons": "ATL",
            "Buffalo Bills": "BUF",
            "Cincinnati Bengals": "CIN",
            "Jacksonville Jaguars": "JAX",
            "New Orleans Saints": "NO",
            "New York Jets": "NYJ",
            "Baltimore Ravens": "BAL",
            "Denver Broncos": "DEN",
            "Houston Texans": "HOU",
            "Indianapolis Colts": "IND",
            "New England Patriots": "NE",
            "Washington Commanders": "WAS",
        }
    )
    TEAM_BYE_WEEKS_2024: Dict[int, List[str]] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Hyperparameters for Reinforcement Learning training."""

    RESUME_TRAINING: bool = True
    TOTAL_EPISODES: int = 1_000_000
    LEARNING_RATE: float = 0.0005
    DISCOUNT_FACTOR: float = 0.99
    VALUE_LOSS_COEFFICIENT: float = 0.5
    ENTROPY_COEFFICIENT: float = 0.01
    BATCH_EPISODES: int = 16
    GRAD_CLIP_NORM: float = 0.5
    VALUE_LR_MULTIPLIER: float = 2.0
    LOG_SAVE_INTERVAL_EPISODES: int = 128
    HIDDEN_DIM: int = 64
    MODEL_PATH_TO_LOAD: str = os.path.join("models/12_teams_pos_5/v1/checkpoint_episode_498.pth")
    NUM_SIMULATION_RUNS: int = 10
    STATE_NORMALIZATION_METHOD: str = "min_max"
    ENABLE_ACTION_MASKING: bool = True
    ALL_STATE_FEATURES: List[str] = field(default_factory=list)
    ENABLED_STATE_FEATURES: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Populate default feature lists when not provided."""
        if not self.ALL_STATE_FEATURES:
            self.ALL_STATE_FEATURES = _default_state_features()
        if not self.ENABLED_STATE_FEATURES:
            self.ENABLED_STATE_FEATURES = _default_enabled_state_features()


@dataclass
class RewardConfig:
    """Settings for reward functions and simulation-based rewards."""

    INVALID_ACTION_PENALTIES: Dict[str, float] = field(
        default_factory=lambda: {
            "already_drafted": -100,
            "roster_full_QB": -50,
            "roster_full_RB": -40,
            "roster_full_WR": -40,
            "roster_full_TE": -50,
            "no_players_available": -100,
            "default_invalid": -50,
        }
    )
    ENABLE_INVALID_ACTION_PENALTIES: bool = False
    ENABLE_INTERMEDIATE_REWARD: bool = False
    INTERMEDIATE_REWARD_MODE: str = "PROPORTIONAL"
    INTERMEDIATE_REWARD_VALUE: float = 30.0
    ENABLE_PICK_SHAPING_REWARD: bool = False
    PICK_SHAPING_STARTER_DELTA_WEIGHT: float = 0.25
    ENABLE_VORP_PICK_SHAPING: bool = False
    VORP_PICK_SHAPING_WEIGHT: float = 0.05
    ENABLE_ROSTER_SLOT_WEIGHTED_REWARD: bool = False
    STARTER_POINTS_WEIGHT: float = 1.0
    BENCH_POINTS_WEIGHT: float = 0.25
    BONUS_FOR_FULL_ROSTER: float = 0.0
    ENABLE_COMPETITIVE_REWARD: bool = False
    COMPETITIVE_REWARD_MODE: str = "SEASON_SIM"
    ENABLE_SEASON_SIM_REWARD: bool = True
    USE_RANDOM_MATCHUPS: bool = True
    NUM_REGULAR_SEASON_WEEKS: int = 14
    REGULAR_SEASON_REWARD: Dict[str, Any] = field(
        default_factory=lambda: {
            "SEED_REWARD_MODE": "LINEAR",
            "NUM_PLAYOFF_TEAMS": 6,
            "MAKE_PLAYOFFS_BONUS": 10.0,
            "SEED_REWARD_MAX": 30.0,
            "SEED_REWARD_MIN": 5.0,
            "SEED_REWARD_MAPPING": {1: 30.0, 2: 24.0, 3: 18.0, 4: 12.0, 5: 8.0, 6: 5.0},
        }
    )
    PLAYOFF_PLACEMENT_REWARDS: Dict[str, float] = field(
        default_factory=lambda: {
            "CHAMPION": 100.0,
            "RUNNER_UP": 40.0,
            "SEMIFINALIST": 15.0,
            "QUARTERFINALIST": 5.0,
            "NON_PLAYOFF": 0.0,
        }
    )
    ENABLE_FINAL_BASE_REWARD: bool = False
    ENABLE_OPPONENT_STD_DEV_PENALTY: bool = False
    OPPONENT_STD_DEV_PENALTY_WEIGHT: float = 0.05
    ENABLE_STACKING_REWARD: bool = True
    STACKING_REWARD_WEIGHT: float = 5.0


@dataclass
class OpponentConfig:
    """Configuration for opponent strategies and personalities."""

    COMPETING_TEAM_LOGIC: str = "HEURISTIC"
    COMPETING_TEAM_RANDOMNESS_FACTOR: float = 0.2
    SUBOPTIMAL_PICK_STRATEGY: str = "NEXT_BEST_ADP"
    DEFAULT_OPPONENT_STRATEGY: Dict[str, Any] = field(
        default_factory=lambda: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.2,
            "suboptimal_strategy": "NEXT_BEST_ADP",
            "positional_priority": ["RB", "WR", "QB", "TE"],
        }
    )
    OPPONENT_TEAM_STRATEGIES: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    OPPONENT_MODEL_PATHS: Dict[str, str] = field(default_factory=dict)
    RANDOMIZE_OPPONENT_STRATEGIES: bool = True
    RANDOMIZE_ONLY_DURING_TRAINING: bool = True
    RANDOMIZE_INCLUDE_AGENT_MODELS: bool = False
    OPPONENT_STRATEGY_TEMPLATES: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataConfig:
    """Configuration for player data generation and nflverse stat loading."""

    LEGACY_STATS_LOOKBACK_SEASONS: int = 2
    NFLVERSE_STATS_RELEASE: str = "stats_player"
    NFLVERSE_WEEK_STATS_TEMPLATE: str = "stats_player_week_{season}.csv"

    def legacy_stats_start_year(self, draft_year: int) -> int:
        """Return the first nflverse season to load for a draft year.

        Parameters
        ----------
        draft_year : int
            Target draft season.

        Returns
        -------
        int
            First season included when fetching weekly stats for veteran
            projections (``draft_year - LEGACY_STATS_LOOKBACK_SEASONS``).
        """
        return draft_year - self.LEGACY_STATS_LOOKBACK_SEASONS


def _default_state_features() -> List[str]:
    """Return the full RL state feature list."""
    return [
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
        "qb_scarcity",
        "rb_scarcity",
        "wr_scarcity",
        "te_scarcity",
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
        "qb_imminent_threat",
        "rb_imminent_threat",
        "wr_imminent_threat",
        "te_imminent_threat",
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
        "current_stack_count",
        "stack_target_available_flag",
    ]


def _default_enabled_state_features() -> List[str]:
    """Return the enabled RL state feature list."""
    return [
        feature
        for feature in _default_state_features()
        if feature
        not in {
            "qb_available_flag",
            "rb_available_flag",
            "wr_available_flag",
            "te_available_flag",
        }
    ]


def _default_opponent_team_strategies() -> Dict[int, Dict[str, Any]]:
    """Return default per-team opponent strategies."""
    return {
        1: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.1,
            "suboptimal_strategy": "NEXT_BEST_HEURISTIC",
            "positional_priority": ["RB", "WR", "QB", "TE"],
        },
        2: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.2,
            "suboptimal_strategy": "NEXT_BEST_HEURISTIC",
            "positional_priority": ["QB", "RB", "WR", "TE"],
        },
        3: {
            "logic": "ADP",
            "randomness_factor": 0.5,
            "suboptimal_strategy": "NEXT_BEST_ADP",
            "positional_priority": ["RB", "WR", "QB", "TE"],
        },
        4: {
            "logic": "ADP",
            "randomness_factor": 0.3,
            "suboptimal_strategy": "NEXT_BEST_ADP",
            "positional_priority": ["WR", "RB", "QB", "TE"],
        },
        5: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.3,
            "suboptimal_strategy": "NEXT_BEST_HEURISTIC",
            "positional_priority": ["WR", "RB", "QB", "TE"],
        },
        6: {
            "logic": "ADP",
            "randomness_factor": 0.7,
            "suboptimal_strategy": "NEXT_BEST_ADP",
            "positional_priority": ["WR", "RB", "QB", "TE"],
        },
        8: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.7,
            "suboptimal_strategy": "NEXT_BEST_HEURISTIC",
            "positional_priority": ["WR", "RB", "QB", "TE"],
        },
        9: {
            "logic": "HEURISTIC",
            "randomness_factor": 0.1,
            "suboptimal_strategy": "NEXT_BEST_HEURISTIC",
            "positional_priority": ["RB", "WR", "QB", "TE"],
        },
    }


def _default_opponent_strategy_templates() -> List[Dict[str, Any]]:
    """Return default opponent randomization templates."""
    return [
        {
            "logic": "HEURISTIC",
            "randomness_factor_range": (0.05, 0.45),
            "suboptimal_strategy_choices": ["NEXT_BEST_HEURISTIC", "NEXT_BEST_ADP"],
            "positional_priority_choices": [
                ["RB", "WR", "QB", "TE"],
                ["WR", "RB", "QB", "TE"],
                ["QB", "RB", "WR", "TE"],
            ],
        },
        {
            "logic": "ADP",
            "randomness_factor_range": (0.1, 0.6),
            "suboptimal_strategy_choices": ["NEXT_BEST_ADP", "RANDOM_ELIGIBLE"],
            "positional_priority_choices": [["RB", "WR", "QB", "TE"], ["WR", "RB", "QB", "TE"]],
        },
        {
            "logic": "RANDOM",
            "randomness_factor_range": (0.0, 1.0),
            "suboptimal_strategy_choices": ["RANDOM_ELIGIBLE"],
            "positional_priority_choices": [["RB", "WR", "QB", "TE"]],
        },
    ]


def _normalize_int_keys(mapping: Dict[Any, Any]) -> Dict[int, Any]:
    """Convert stringified integer keys from JSON into integer keys."""
    return {int(key): value for key, value in mapping.items()}


def _coerce_subconfig_value(sub_config: Any, sub_key: str, sub_value: Any) -> Any:
    """Normalize JSON-loaded values before applying them to a sub-config."""
    if sub_key in _INT_KEYED_DRAFT_FIELDS and isinstance(sub_value, dict):
        return _normalize_int_keys(sub_value)
    if sub_key in _INT_KEYED_OPPONENT_FIELDS and isinstance(sub_value, dict):
        return _normalize_int_keys(sub_value)
    if sub_key == "bye_weeks" and isinstance(sub_value, dict):
        return {int(week): teams for week, teams in sub_value.items()}
    if sub_key == "SEED_REWARD_MAPPING" and isinstance(sub_value, dict):
        return {int(seed): points for seed, points in sub_value.items()}
    return sub_value


class Config:
    """Main configuration class that aggregates all sub-configs."""

    def __init__(self) -> None:
        self.league = LeagueMetaConfig()
        self.paths = PathsConfig()
        self.draft = DraftConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.reward = RewardConfig()
        self.opponent = OpponentConfig()
        self.scoring = ScoringConfig()
        self.season = SeasonRuntimeConfig()
        if not self.opponent.OPPONENT_TEAM_STRATEGIES:
            self.opponent.OPPONENT_TEAM_STRATEGIES = _default_opponent_team_strategies()
        if not self.opponent.OPPONENT_STRATEGY_TEMPLATES:
            self.opponent.OPPONENT_STRATEGY_TEMPLATES = _default_opponent_strategy_templates()

    def get_scoring_rules(self) -> Dict[str, Optional[float]]:
        """Return the active fantasy scoring rules for this league."""
        return dict(self.scoring.rules)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary."""
        return {
            "league": asdict(self.league),
            "paths": asdict(self.paths),
            "draft": asdict(self.draft),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "reward": asdict(self.reward),
            "opponent": asdict(self.opponent),
            "scoring": asdict(self.scoring),
            "season": asdict(self.season),
        }

    def save(self, filepath: str) -> None:
        """Save the configuration to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        instance = cls()
        for key, value in data.items():
            if not hasattr(instance, key) or not isinstance(value, dict):
                continue
            sub_config = getattr(instance, key)
            for sub_key, sub_value in value.items():
                setattr(sub_config, sub_key, _coerce_subconfig_value(sub_config, sub_key, sub_value))
        return instance

    @classmethod
    def from_file(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return cls.from_dict(data)
