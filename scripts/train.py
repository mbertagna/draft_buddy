import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv
from reinforce_agent import ReinforceAgent
from policy_network import PolicyNetwork # Not directly instantiated here, but useful for context if loading a model

def main():
    """
    Main function to initialize the environment, agent, and run the training process.
    """
    print("--- Starting Fantasy Football Draft AI Training ---")

    # 1. Load Configuration
    config = Config()
    print(f"Project Base Directory: {config.BASE_DIR}")
    print(f"Data CSV: {config.PLAYER_DATA_CSV}")
    print(f"Total Episodes: {config.TOTAL_EPISODES}")
    print(f"Agent Start Position: {config.AGENT_START_POSITION}")
    print(f"State Normalization: {config.STATE_NORMALIZATION_METHOD}")
    print(f"Intermediate Reward Enabled: {config.ENABLE_INTERMEDIATE_REWARD}")

    # 2. Initialize Environment
    print("\nInitializing Fantasy Football Draft Environment...")
    env = FantasyFootballDraftEnv(config)
    print(f"Observation Space Dimension: {env.observation_space_dim}")
    print(f"Action Space: {env.action_space.n} actions ({env.action_to_position})")

    # 3. Initialize Agent
    print("\nInitializing REINFORCE Agent...")
    agent = ReinforceAgent(env, config)
    print(f"Policy Network Architecture: Input({len(config.ENABLED_STATE_FEATURES)}) -> Hidden({config.HIDDEN_DIM}) -> Hidden({config.HIDDEN_DIM}) -> Output({env.action_space.n})")
    print(f"Learning Rate: {config.LEARNING_RATE}, Discount Factor (Gamma): {config.DISCOUNT_FACTOR}")

    # 4. Train the Agent
    print("\nStarting Agent Training...")
    episode_rewards, policy_losses = agent.train()
    print("\nAgent training complete!")

    # 5. Save the trained model
    model_save_path = os.path.join(config.MODELS_DIR, f"reinforce_policy_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(agent.policy_network.state_dict(), model_save_path)
    print(f"\nPolicy network model saved to: {model_save_path}")

    # 6. Plotting Training Results
    print("\nGenerating training plots...")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    rewards_plot_path = os.path.join(config.LOGS_DIR, f"rewards_plot_{timestamp}.png")
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
    losses_plot_path = os.path.join(config.LOGS_DIR, f"losses_plot_{timestamp}.png")
    plt.savefig(losses_plot_path)
    plt.close()
    print(f"Losses plot saved to: {losses_plot_path}")
    
    print("\nTraining process finished.")
    print("You can inspect the saved model and plots in the 'models/' and 'logs/' directories.")

if __name__ == "__main__":
    main()