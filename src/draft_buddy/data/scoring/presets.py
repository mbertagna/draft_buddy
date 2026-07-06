"""Named scoring rule presets for tests and league profile documentation."""

from typing import Dict, Optional

ESPN_FULL_PPR_TRACKABLE: Dict[str, Optional[float]] = {
    "passing_yards": 0.04,
    "passing_tds": 6,
    "interceptions": -2,
    "passing_2pt_conversions": 2,
    "passing_yards_300_399_game": 2,
    "passing_yards_400_plus_game": 6,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "rushing_2pt_conversions": 2,
    "rushing_yards_100_199_game": 3,
    "rushing_yards_200_plus_game": 6,
    "receptions": 1.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "receiving_2pt_conversions": 2,
    "receiving_yards_100_199_game": 3,
    "receiving_yards_200_plus_game": 6,
    "total_fumbles_lost": -2,
    "pat_made": 1,
    "pat_missed": -1,
    "fg_missed": -1,
    "fg_made_0_39": 3,
    "fg_made_40_49": 4,
    "fg_made_50_59": 5,
    "fg_made_60_": 6,
}

SLEEPER_HALF_PPR_TRACKABLE: Dict[str, Optional[float]] = {
    "passing_yards": 0.04,
    "passing_tds": 4,
    "interceptions": -1,
    "passing_2pt_conversions": 2,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "rushing_2pt_conversions": 2,
    "receptions": 0.5,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "receiving_2pt_conversions": 2,
    "fg_made_yards": 0.1,
    "xp_made": 1,
    "xp_missed": -1,
    "total_fumbles_lost": -2,
}
