import os
import json
import glob
from datetime import datetime
import re


def get_run_name(config):
    """Generate a descriptive training run name from config."""
    if getattr(config.draft, "RANDOMIZE_AGENT_START_POSITION", False):
        return f"{config.draft.NUM_TEAMS}_teams_random_start"
    return f"{config.draft.NUM_TEAMS}_teams_pos_{config.draft.AGENT_START_POSITION}"


def get_next_version(run_dir):
    """Find the next run version identifier such as v1 or v2."""
    if not os.path.exists(run_dir):
        return "v1"
    existing_versions = [
        directory
        for directory in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, directory)) and directory.startswith("v")
    ]
    if not existing_versions:
        return "v1"
    latest_version = max(int(version.replace("v", "")) for version in existing_versions)
    return f"v{latest_version + 1}"


def setup_run_directories(config):
    """Create models and logs directories for a new run version."""
    run_name = get_run_name(config)
    base_run_dir = os.path.join(config.paths.MODELS_DIR, run_name)
    version = get_next_version(base_run_dir)
    run_version_dir = os.path.join(base_run_dir, version)
    logs_dir = os.path.join(config.paths.LOGS_DIR, run_name, version)
    os.makedirs(run_version_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return run_name, version, run_version_dir, logs_dir


def save_run_metadata(config, run_name, version, run_version_dir):
    """Persist run metadata and config snapshot into metadata.json."""
    metadata = {
        "run_name": run_name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "config": (
            config.to_dict()
            if hasattr(config, "to_dict")
            else {
                key: value
                for key, value in config.__class__.__dict__.items()
                if not key.startswith("__") and isinstance(value, (int, float, str, bool, list, dict))
            }
        ),
    }
    metadata_path = os.path.join(run_version_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=4)
    print(f"Run metadata saved to {metadata_path}")


def find_latest_checkpoint(config):
    """
    Find latest checkpoint path for current run name using highest episode number.
    """
    run_name = get_run_name(config)
    base_run_dir = os.path.join(config.paths.MODELS_DIR, run_name)
    if not os.path.exists(base_run_dir):
        return None
    checkpoints = glob.glob(
        os.path.join(base_run_dir, "**", "checkpoint_episode_*.pth"), recursive=True
    )
    if not checkpoints:
        print(f"No checkpoint files found for run '{run_name}'.")
        return None
    latest_checkpoint = None
    latest_episode_num = -1
    for checkpoint_path in checkpoints:
        match = re.search(r"checkpoint_episode_(\d+)\.pth$", os.path.basename(checkpoint_path))
        if not match:
            continue
        episode_num = int(match.group(1))
        if episode_num > latest_episode_num:
            latest_episode_num = episode_num
            latest_checkpoint = checkpoint_path
    if latest_checkpoint:
        print(f"Found latest checkpoint (episode {latest_episode_num}): {latest_checkpoint}")
    else:
        print(f"No checkpoint files with episode numbers found for run '{run_name}'.")
    return latest_checkpoint
