import pytest
import os
import tempfile
import shutil
from draft_buddy.config import Config
from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.rules_engine import FantasyRulesEngine
from draft_buddy.domain.entities import Player

@pytest.fixture(scope="session")
def test_env_setup():
    """
    Sets up a completely isolated environment for the duration of the test session.
    Creates temporary directories for data, models, and logs.
    """
    # Create a temporary root for the test run
    test_root = tempfile.mkdtemp()
    
    # Define subdirectories
    dirs = {
        'data': os.path.join(test_root, 'data'),
        'models': os.path.join(test_root, 'models'),
        'logs': os.path.join(test_root, 'logs')
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # Create a dummy player data file for tests
    player_csv = os.path.join(dirs['data'], 'generated_player_data.csv')
    csv_content = (
        "player_id,name,position,projected_points,adp,games_played_frac\n"
        "1,Test QB,QB,300.0,10.0,1.0\n"
        "2,Test RB,RB,200.0,20.0,1.0\n"
        "3,Test WR,WR,150.0,30.0,1.0\n"
        "4,Test TE,TE,100.0,40.0,1.0\n"
    )
    with open(player_csv, 'w') as f:
        f.write(csv_content)

    yield dirs, player_csv

    # Cleanup after all tests are done
    shutil.rmtree(test_root)

@pytest.fixture
def mock_config(test_env_setup):
    """
    Provides a Config object that points to temporary test directories.
    """
    dirs, player_csv = test_env_setup
    config = Config()
    
    # Override paths to use temporary test directories
    config.paths.DATA_DIR = dirs['data']
    config.paths.MODELS_DIR = dirs['models']
    config.paths.LOGS_DIR = dirs['logs']
    config.paths.PLAYER_DATA_CSV = player_csv
    
    return config


@pytest.fixture
def core_roster_structure():
    """Return a standard starter roster structure.

    Returns
    -------
    dict
        Starter slots by position.
    """
    return {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}


@pytest.fixture
def core_bench_maxes():
    """Return bench limits used for core tests.

    Returns
    -------
    dict
        Bench slots by position.
    """
    return {"QB": 1, "RB": 2, "WR": 2, "TE": 1}


@pytest.fixture
def core_total_roster_size(core_roster_structure, core_bench_maxes):
    """Return total roster size derived from starters and bench.

    Parameters
    ----------
    core_roster_structure : dict
        Starter slot counts.
    core_bench_maxes : dict
        Bench slot counts.

    Returns
    -------
    int
        Total roster cap.
    """
    return sum(core_roster_structure.values()) + sum(core_bench_maxes.values())


@pytest.fixture
def core_player_pool():
    """Return a small deterministic player pool.

    Returns
    -------
    list[Player]
        Players across core fantasy positions.
    """
    return [
        Player(1, "QB One", "QB", 300.0, adp=10.0, team="BUF"),
        Player(2, "RB One", "RB", 220.0, adp=12.0, team="BUF"),
        Player(3, "RB Two", "RB", 210.0, adp=18.0, team="MIA"),
        Player(4, "RB Three", "RB", 190.0, adp=32.0, team="NYJ"),
        Player(5, "WR One", "WR", 230.0, adp=9.0, team="BUF"),
        Player(6, "WR Two", "WR", 180.0, adp=26.0, team="MIA"),
        Player(7, "WR Three", "WR", 170.0, adp=34.0, team="NE"),
        Player(8, "WR Four", "WR", 160.0, adp=40.0, team="PIT"),
        Player(9, "TE One", "TE", 150.0, adp=50.0, team="BUF"),
        Player(10, "TE Two", "TE", 130.0, adp=70.0, team="NYG"),
    ]


@pytest.fixture
def core_player_map(core_player_pool):
    """Return player map keyed by player id.

    Parameters
    ----------
    core_player_pool : list[Player]
        Source players.

    Returns
    -------
    dict[int, Player]
        Lookup table for players.
    """
    return {player.player_id: player for player in core_player_pool}


@pytest.fixture
def core_draft_state(core_player_map, core_roster_structure, core_bench_maxes, core_total_roster_size):
    """Return a fresh DraftState for core tests.

    Parameters
    ----------
    core_player_map : dict[int, Player]
        Players available in the draft pool.
    core_roster_structure : dict
        Starter slot counts.
    core_bench_maxes : dict
        Bench slot counts.
    core_total_roster_size : int
        Total roster cap.

    Returns
    -------
    DraftState
        Initialized mutable draft state.
    """
    return DraftState(
        all_player_ids=set(core_player_map.keys()),
        draft_order=[1, 2, 3, 4],
        roster_structure=core_roster_structure,
        bench_maxes=core_bench_maxes,
        total_roster_size_per_team=core_total_roster_size,
    )


@pytest.fixture
def core_rules_engine(core_roster_structure, core_bench_maxes, core_total_roster_size):
    """Return the fantasy rules engine for core tests.

    Parameters
    ----------
    core_roster_structure : dict
        Starter slot counts.
    core_bench_maxes : dict
        Bench slot counts.
    core_total_roster_size : int
        Total roster cap.

    Returns
    -------
    FantasyRulesEngine
        Rules engine with deterministic limits.
    """
    return FantasyRulesEngine(
        roster_structure=core_roster_structure,
        bench_maxes=core_bench_maxes,
        total_roster_size_per_team=core_total_roster_size,
    )
