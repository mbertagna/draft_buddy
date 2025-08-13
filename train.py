import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from reinforce_agent import ReinforceAgent
from utils.run_utils import setup_run_directories, save_run_metadata, find_latest_checkpoint

def plot_training_results(episode_rewards, policy_losses, logs_dir, prefix=""):
    """
    Plots training results and saves them to the specified directory.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot Total Rewards
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

    # Plot Policy Losses
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
