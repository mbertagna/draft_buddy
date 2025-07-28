import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from reinforce_agent import ReinforceAgent
from utils.run_utils import setup_run_directories, save_run_metadata, find_latest_checkpoint

def main():
    """
    Main function to initialize the environment, agent, and run the training process.
    """
    print("--- Starting Fantasy Football Draft AI Training ---")

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
        else:
            print("\nNo checkpoint found. Starting a new training run.")

    # 6. Train the Agent
    print("\nStarting Agent Training...")
    episode_rewards, policy_losses = agent.train(start_episode=start_episode, run_version_dir=run_version_dir)
    print("\nAgent training complete!")

    # 7. Plotting Training Results
    print("\nGenerating training plots...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    rewards_plot_path = os.path.join(logs_dir, f"rewards_plot_{timestamp}.png")
    plt.savefig(rewards_plot_path)
    plt.close()
    print(f"Rewards plot saved to: {rewards_plot_path}")

    # Plot Policy Losses
    plt.figure(figsize=(12, 6))
    plt.plot(policy_losses)
    plt.title('Policy Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    losses_plot_path = os.path.join(logs_dir, f"losses_plot_{timestamp}.png")
    plt.savefig(losses_plot_path)
    plt.close()
    print(f"Losses plot saved to: {losses_plot_path}")
    
    print("\nTraining process finished.")
    print(f"You can inspect the saved model and plots in the '{run_version_dir}' and '{logs_dir}' directories.")

if __name__ == "__main__":
    main()
