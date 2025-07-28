import os
import json
import glob
from datetime import datetime

def get_run_name(config):
    """Generates a descriptive name for the training run based on config."""
    return f"{config.NUM_TEAMS}_teams_pos_{config.AGENT_START_POSITION}"

def get_next_version(run_dir):
    """Finds the next version number (e.g., v1, v2) for a given run."""
    if not os.path.exists(run_dir):
        return "v1"
    
    existing_versions = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith('v')]
    if not existing_versions:
        return "v1"
    
    latest_version = max(int(v.replace('v', '')) for v in existing_versions)
    return f"v{latest_version + 1}"

def setup_run_directories(config):
    """Creates the directory structure for a new training run or version."""
    run_name = get_run_name(config)
    base_run_dir = os.path.join(config.MODELS_DIR, run_name)
    
    version = get_next_version(base_run_dir)
    
    run_version_dir = os.path.join(base_run_dir, version)
    logs_dir = os.path.join(config.LOGS_DIR, run_name, version)
    
    os.makedirs(run_version_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return run_name, version, run_version_dir, logs_dir

def save_run_metadata(config, run_name, version, run_version_dir):
    """Saves the configuration of the run to a metadata.json file."""
    metadata = {
        'run_name': run_name,
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'config': {k: v for k, v in config.__class__.__dict__.items() if not k.startswith('__') and isinstance(v, (int, float, str, bool, list, dict))}
    }
    
    metadata_path = os.path.join(run_version_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Run metadata saved to {metadata_path}")

def find_latest_checkpoint(config):
    """Finds the latest model checkpoint from the previous version of a run."""
    run_name = get_run_name(config)
    base_run_dir = os.path.join(config.MODELS_DIR, run_name)

    if not os.path.exists(base_run_dir):
        return None

    existing_versions = [d for d in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, d)) and d.startswith('v')]
    if not existing_versions:
        return None

    version_nums = [int(v.replace('v', '')) for v in existing_versions]
    if not version_nums or max(version_nums) < 2:
        # Cannot resume if there's no version before the current one.
        return None

    # The current version is the max, so the previous version is max - 1.
    previous_version_num = max(version_nums) - 1
    previous_version_dir = os.path.join(base_run_dir, f"v{previous_version_num}")

    if not os.path.exists(previous_version_dir):
        return None

    # Look specifically for checkpoint files.
    checkpoints = glob.glob(os.path.join(previous_version_dir, 'checkpoint_episode_*.pth'))
    if not checkpoints:
        return None

    # Return the path to the most recently modified checkpoint file.
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint
