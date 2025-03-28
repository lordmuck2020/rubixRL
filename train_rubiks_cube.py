import gymnasium as gym
from rubixRL.agents.ppo_agent import PPOAgent
import numpy as np
import torch
import os
from datetime import datetime


def main():
    # Create environment
    env = gym.make("RubiksCube-v0", n=3, max_steps=100, reward_type="dense")

    # Create agent
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=1.0,
        c2=0.01,
        batch_size=64,
        n_epochs=10,
        hidden_dim=256,
    )

    # Create directory for saving models
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Training parameters
    n_episodes = 1000
    max_steps = 100
    save_interval = 100  # Save model every 100 episodes

    # Training loop
    print("Starting training...")
    episode_rewards = agent.train(n_episodes=n_episodes, max_steps=max_steps)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"rubiks_cube_ppo_{timestamp}.pt")
    agent.save(save_path)
    print(f"Training completed. Model saved to {save_path}")

    # Print final statistics
    print("\nTraining Statistics:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Standard Deviation: {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
