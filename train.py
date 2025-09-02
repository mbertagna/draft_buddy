import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from reinforce_agent import ReinforceAgent
from utils.run_utils import setup_run_directories, save_run_metadata, find_latest_checkpoint, get_run_name

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool
from bokeh.layouts import column

def plot_training_results(episode_rewards, policy_losses, logs_dir, prefix=""):
    """
    Plots training results as static PNGs and combined interactive Bokeh HTML dashboard.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --------------------
    # Matplotlib (PNG)
    # --------------------
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title(f'{prefix}Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
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

    # --------------------
    # Bokeh (Interactive HTML Dashboard)
    # --------------------
    html_path = os.path.join(logs_dir, f"{prefix}training_dashboard_{timestamp}.html")
    output_file(html_path, title=f"{prefix} Training Dashboard")

    # Explicit x values (lists instead of range objects)
    x_rewards = list(range(len(episode_rewards)))
    x_losses = list(range(len(policy_losses)))

    # Shared x_range for synchronized zooming
    x_range = (0, max(len(episode_rewards), len(policy_losses)))

    p1 = figure(
        title=f"{prefix}Total Reward per Episode",
        x_axis_label="Episode",
        y_axis_label="Total Reward",
        width=1000,
        height=400,
        tools="pan,xwheel_zoom,reset,save",  # only zooms x with wheel
        active_scroll="xwheel_zoom",         # make x-zoom active
        x_range=x_range
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
        tools="pan,xwheel_zoom,reset,save",  # only zooms x with wheel
        active_scroll="xwheel_zoom",
        x_range=p1.x_range  # sync x zoom
    )
    p2.line(x_losses, policy_losses, line_width=2, color="red", legend_label="Loss")
    p2.add_tools(HoverTool(tooltips=[("Episode", "$x"), ("Loss", "$y")]))
    p2.legend.click_policy = "hide"

    layout = column(p1, p2)
    save(layout)

    print(f"Interactive training dashboard saved to: {html_path}")

def find_latest_logs_dir_with_csvs(logs_root: str) -> str:
    """Recursively search for the most recently updated logs directory containing both CSVs."""
    candidates = []
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

def find_version_dirs_with_csvs(run_logs_root: str):
    """Return a list of (version_num, dir_path) for version dirs under run_logs_root that contain both CSVs."""
    results = []
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

def _load_floats_from_csv(path: str):
    with open(path, "r") as f:
        return [float(line.strip()) for line in f if line.strip()]

def main():
    """
    Main function to initialize the environment, agent, and run the training process.
    """
    print("--- Starting Fantasy Football Draft AI Training ---")

    # CLI args
    parser = argparse.ArgumentParser(description="Train agent or plot latest CSV results.")
    parser.add_argument(
        "--plot-latest-csvs", 
        "-p",
        action="store_true",
        help="Auto-find the latest logs directory with CSVs and generate plots without training.",
    )
    args = parser.parse_args()

    # If only plotting is requested, auto-find logs dir and plot, then exit
    if args.plot_latest_csvs:
        config = Config()
        # Aggregate across all versions under the run's logs root
        run_name = get_run_name(config)
        run_logs_root = os.path.join(config.LOGS_DIR, run_name)
        version_dirs = find_version_dirs_with_csvs(run_logs_root)
        if not version_dirs:
            # fallback to legacy behavior of searching entire logs root
            latest_logs_dir = find_latest_logs_dir_with_csvs(config.LOGS_DIR)
            if not latest_logs_dir:
                print(f"No logs directory with both CSVs found under '{config.LOGS_DIR}'.")
                return
            rewards_csv = os.path.join(latest_logs_dir, "all_episode_rewards.csv")
            losses_csv = os.path.join(latest_logs_dir, "all_policy_losses.csv")
            if not (os.path.exists(rewards_csv) and os.path.exists(losses_csv)):
                print(f"Missing CSVs in '{latest_logs_dir}'. Expected both rewards and losses CSVs.")
                return
            print(f"Auto-found logs directory: {latest_logs_dir}")
            episode_rewards = _load_floats_from_csv(rewards_csv)
            policy_losses = _load_floats_from_csv(losses_csv)
            print("Generating plots from discovered CSVs...")
            plot_training_results(episode_rewards, policy_losses, latest_logs_dir, prefix="manual_")
            print("Finished plotting from CSVs. Exiting.")
            return

        # Concatenate data from all versions in order
        all_rewards, all_losses = [], []
        for vnum, vdir in version_dirs:
            rewards_csv = os.path.join(vdir, "all_episode_rewards.csv")
            losses_csv = os.path.join(vdir, "all_policy_losses.csv")
            try:
                all_rewards.extend(_load_floats_from_csv(rewards_csv))
            except Exception:
                pass
            try:
                all_losses.extend(_load_floats_from_csv(losses_csv))
            except Exception:
                pass

        # Save aggregate plots in the run root logs dir
        print("Auto-found run logs root:", run_logs_root)
        print("Generating aggregated plots from all versions...")
        plot_training_results(all_rewards, all_losses, run_logs_root, prefix="manual_aggregate_")
        print("Finished plotting aggregated CSVs. Exiting.")
        return

    # 1. Load Configuration
    config = Config()

    # 2. Setup Run Directories and Metadata
    run_name, version, run_version_dir, logs_dir = setup_run_directories(config)
    save_run_metadata(config, run_name, version, run_version_dir)

    print(f"Run: {run_name} | Version: {version}")
    print(f"Models will be saved in: {run_version_dir}")
    print(f"Logs will be saved in: {logs_dir}")

    # 3. Initialize Environment
    print("\nInitializing Fantasy Football Draft Environment...")
    env = FantasyFootballDraftEnv(config, training=True)

    # 4. Initialize Agent
    print("\nInitializing REINFORCE Agent...")
    agent = ReinforceAgent(env, config)

    # 5. Load Checkpoint if Resuming
    start_episode = 1
    if config.RESUME_TRAINING:
        latest_checkpoint = find_latest_checkpoint(config)
        if latest_checkpoint:
            print(f"\nResuming training from checkpoint: {latest_checkpoint}")
            agent.load_model(latest_checkpoint)
            # Extract episode number from checkpoint filename if possible
            try:
                base_name = os.path.basename(latest_checkpoint)
                start_episode = int(base_name.split('episode_')[1].split('.')[0]) + 1
                print(f"Starting from episode {start_episode}")
            except (IndexError, ValueError):
                print("Could not determine start episode from checkpoint filename. Starting from episode 1.")

            # Load existing raw data for plotting
            rewards_data_path = os.path.join(logs_dir, 'all_episode_rewards.csv')
            losses_data_path = os.path.join(logs_dir, 'all_policy_losses.csv')
            if os.path.exists(rewards_data_path):
                with open(rewards_data_path, 'r') as f:
                    agent.all_episode_rewards = [float(line.strip()) for line in f]
            if os.path.exists(losses_data_path):
                with open(losses_data_path, 'r') as f:
                    agent.all_policy_losses = [float(line.strip()) for line in f]

            # Try to resume optimizer and value network if available
            optimizer_path_guess = os.path.join(os.path.dirname(latest_checkpoint), f"optimizer_episode_{start_episode-1}.pt")
            value_path_guess = os.path.join(os.path.dirname(latest_checkpoint), f"value_episode_{start_episode-1}.pth")
            if os.path.exists(optimizer_path_guess):
                try:
                    agent.load_optimizer(optimizer_path_guess)
                except Exception as e:
                    print(f"Warning: failed to load optimizer from {optimizer_path_guess}: {e}")
            if os.path.exists(value_path_guess):
                try:
                    agent.load_value_network(value_path_guess)
                except Exception as e:
                    print(f"Warning: failed to load value network from {value_path_guess}: {e}")

        else:
            print("\nNo checkpoint found. Starting a new training run.")

    # 6. Train the Agent
    print("\nStarting Agent Training...")
    episode_rewards, policy_losses = agent.train(start_episode=start_episode, run_version_dir=run_version_dir, logs_dir=logs_dir)
    print("\nAgent training complete!")

    # 7. Plotting Training Results (Final)
    print("\nGenerating final training plots...")
    plot_training_results(episode_rewards, policy_losses, logs_dir, prefix="final_")
    
    print("\nTraining process finished.")
    print(f"You can inspect the saved model and plots in the '{run_version_dir}' and '{logs_dir}' directories.")

if __name__ == "__main__":
    main()
