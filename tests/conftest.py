import pytest
import os
import tempfile
import shutil
from draft_buddy.config import Config

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
