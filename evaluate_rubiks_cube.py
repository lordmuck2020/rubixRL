import gymnasium as gym
from rubixRL.agents.ppo_agent import PPOAgent
import numpy as np
import torch
import os
from datetime import datetime


def evaluate_agent(agent: PPOAgent, n_episodes: int = 10, render: bool = True):
    """
    Evaluate the trained agent.

    Args:
        agent: The trained PPO agent
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    env = agent.env
    episode_rewards = []
    episode_steps = []
    solved_count = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            if render:
                env.render()

            # Select action (no exploration during evaluation)
            action, _, _ = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

        if info.get("is_solved", False):
            solved_count += 1

    # Print evaluation statistics
    print("\nEvaluation Statistics:")
    print(f"Number of Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f}")
    print(f"Success Rate: {(solved_count / n_episodes) * 100:.1f}%")

    return episode_rewards, episode_steps


def main():
    # Create environment
    env = gym.make("RubiksCube-v0", n=3, max_steps=100, reward_type="dense")

    # Create agent
    agent = PPOAgent(env=env)

    # Load the trained model
    model_path = (
        "models/rubiks_cube_ppo_latest.pt"  # Update this path to your trained model
    )
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"No trained model found at {model_path}")
        return

    # Evaluate the agent
    print("Starting evaluation...")
    episode_rewards, episode_steps = evaluate_agent(agent, n_episodes=10, render=True)


if __name__ == "__main__":
    main()
