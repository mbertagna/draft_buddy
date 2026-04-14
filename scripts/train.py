import argparse
import datetime
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from draft_buddy.config import Config
from draft_buddy.rl.reinforce_agent import ReinforceAgent
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool
from bokeh.layouts import column
from draft_buddy.rl import GymEnv
from draft_buddy.rl.run_utils import (
    find_latest_checkpoint,
    get_run_name,
    save_run_metadata,
    setup_run_directories,
)


def plot_training_results(
    episode_rewards: List[float], policy_losses: List[float], logs_dir: str, prefix: str = ""
) -> None:
    """Plot and save reward/loss trends.

    Parameters
    ----------
    episode_rewards : List[float]
        Episode reward series.
    policy_losses : List[float]
        Policy loss series.
    logs_dir : str
        Output directory for plots.
    prefix : str, optional
        Filename prefix.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title(f"{prefix}Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    rewards_plot_path = os.path.join(logs_dir, f"{prefix}rewards_plot_{timestamp}.png")
    plt.savefig(rewards_plot_path)
    plt.close()
    print(f"Rewards plot saved to: {rewards_plot_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(policy_losses)
    plt.title(f'{prefix}Policy Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    losses_plot_path = os.path.join(logs_dir, f"{prefix}losses_plot_{timestamp}.png")
    plt.savefig(losses_plot_path)
    plt.close()
    print(f"Losses plot saved to: {losses_plot_path}")

    html_path = os.path.join(logs_dir, f"{prefix}training_dashboard_{timestamp}.html")
    output_file(html_path, title=f"{prefix} Training Dashboard")

    x_rewards = list(range(len(episode_rewards)))
    x_losses = list(range(len(policy_losses)))
    x_range = (0, max(len(episode_rewards), len(policy_losses)))

    p1 = figure(
        title=f"{prefix}Total Reward per Episode",
        x_axis_label="Episode",
        y_axis_label="Total Reward",
        width=1000,
        height=400,
        tools="pan,xwheel_zoom,reset,save",
        active_scroll="xwheel_zoom",
        x_range=x_range,
    )
    p1.line(x_rewards, episode_rewards, line_width=2, legend_label="Reward")
    p1.add_tools(HoverTool(tooltips=[("Episode", "$x"), ("Reward", "$y")]))
    p1.legend.click_policy = "hide"

    p2 = figure(
        title=f"{prefix}Policy Loss per Episode",
        x_axis_label="Episode",
        y_axis_label="Loss",
        width=1000,
        height=400,
        tools="pan,xwheel_zoom,reset,save",
        active_scroll="xwheel_zoom",
        x_range=p1.x_range,
    )
    p2.line(x_losses, policy_losses, line_width=2, color="red", legend_label="Loss")
    p2.add_tools(HoverTool(tooltips=[("Episode", "$x"), ("Loss", "$y")]))
    p2.legend.click_policy = "hide"

    layout = column(p1, p2)
    save(layout)

    print(f"Interactive training dashboard saved to: {html_path}")


def find_latest_logs_dir_with_csvs(logs_root: str) -> Optional[str]:
    """Find most recently updated logs directory with both CSV files.

    Parameters
    ----------
    logs_root : str
        Root logs directory.

    Returns
    -------
    Optional[str]
        Matching directory path, if any.
    """
    candidates: List[Tuple[float, str]] = []
    for root, _dirs, files in os.walk(logs_root):
        if {"all_episode_rewards.csv", "all_policy_losses.csv"}.issubset(set(files)):
            try:
                csv_paths = [
                    os.path.join(root, "all_episode_rewards.csv"),
                    os.path.join(root, "all_policy_losses.csv"),
                ]
                mtime = max(os.path.getmtime(p) for p in csv_paths)
            except FileNotFoundError:
                continue
            candidates.append((mtime, root))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_version_dirs_with_csvs(run_logs_root: str) -> List[Tuple[int, str]]:
    """List versioned run directories containing both metrics CSVs.

    Parameters
    ----------
    run_logs_root : str
        Logs root for a run family.

    Returns
    -------
    List[Tuple[int, str]]
        Version number and directory pairs.
    """
    results: List[Tuple[int, str]] = []
    if not os.path.exists(run_logs_root):
        return results
    for entry in os.listdir(run_logs_root):
        dir_path = os.path.join(run_logs_root, entry)
        if not os.path.isdir(dir_path):
            continue
        if not entry.startswith('v'):
            continue
        try:
            version_num = int(entry.replace('v', ''))
        except ValueError:
            continue
        rewards_csv = os.path.join(dir_path, "all_episode_rewards.csv")
        losses_csv = os.path.join(dir_path, "all_policy_losses.csv")
        if os.path.exists(rewards_csv) and os.path.exists(losses_csv):
            results.append((version_num, dir_path))
    results.sort(key=lambda x: x[0])
    return results


def _load_floats_from_csv(path: str) -> List[float]:
    """Load one float per line from CSV-like file."""
    with open(path, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]


def _run_plot_only_mode(config: Config) -> None:
    """Generate plots from latest available CSV metrics files."""
    run_name = get_run_name(config)
    run_logs_root = os.path.join(config.paths.LOGS_DIR, run_name)
    version_dirs = find_version_dirs_with_csvs(run_logs_root)
    if not version_dirs:
        latest_logs_dir = find_latest_logs_dir_with_csvs(config.paths.LOGS_DIR)
        if not latest_logs_dir:
            print(f"No logs directory with both CSVs found under '{config.paths.LOGS_DIR}'.")
            return
        rewards_csv = os.path.join(latest_logs_dir, "all_episode_rewards.csv")
        losses_csv = os.path.join(latest_logs_dir, "all_policy_losses.csv")
        if not (os.path.exists(rewards_csv) and os.path.exists(losses_csv)):
            print(f"Missing CSVs in '{latest_logs_dir}'.")
            return
        episode_rewards = _load_floats_from_csv(rewards_csv)
        policy_losses = _load_floats_from_csv(losses_csv)
        plot_training_results(episode_rewards, policy_losses, latest_logs_dir, prefix="manual_")
        return

    all_rewards: List[float] = []
    all_losses: List[float] = []
    for _, version_dir in version_dirs:
        rewards_csv = os.path.join(version_dir, "all_episode_rewards.csv")
        losses_csv = os.path.join(version_dir, "all_policy_losses.csv")
        all_rewards.extend(_load_floats_from_csv(rewards_csv))
        all_losses.extend(_load_floats_from_csv(losses_csv))
    plot_training_results(all_rewards, all_losses, run_logs_root, prefix="manual_aggregate_")


def main() -> None:
    """Run RL training for Draft Buddy."""
    parser = argparse.ArgumentParser(description="Train agent or plot latest CSV results.")
    parser.add_argument(
        "--plot-latest-csvs",
        "-p",
        action="store_true",
        help="Auto-find the latest logs directory with CSVs and generate plots without training.",
    )
    args = parser.parse_args()

    print("--- Starting Fantasy Football Draft AI Training ---")
    if args.plot_latest_csvs:
        config = Config()
        _run_plot_only_mode(config)
        return

    config = Config()
    run_name, version, run_version_dir, logs_dir = setup_run_directories(config)
    save_run_metadata(config, run_name, version, run_version_dir)
    print(f"Run: {run_name} | Version: {version}")
    print(f"Models will be saved in: {run_version_dir}")
    print(f"Logs will be saved in: {logs_dir}")
    env = GymEnv(config, training=True)
    agent = ReinforceAgent(env, config)
    start_episode = 1
    if config.training.RESUME_TRAINING:
        latest_checkpoint_path = find_latest_checkpoint(config)
        if latest_checkpoint_path:
            print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            try:
                loaded_episode = agent.load_checkpoint(latest_checkpoint_path, is_training=True)
                start_episode = loaded_episode + 1
                print(f"Checkpoint loaded. Resuming from episode {start_episode}")
            except (ValueError, FileNotFoundError) as e:
                print(f"Failed to resume from checkpoint: {e}")
                return
            except Exception as e:
                print(f"Unexpected checkpoint loading failure: {e}")
                return
    episode_rewards, policy_losses = agent.train(
        start_episode=start_episode, run_version_dir=run_version_dir, logs_dir=logs_dir
    )
    plot_training_results(episode_rewards, policy_losses, logs_dir, prefix="final_")
    print("Training finished.")

if __name__ == "__main__":
    main()
