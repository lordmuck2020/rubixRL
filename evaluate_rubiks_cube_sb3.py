import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import numpy as np

# Import the Rubik's Cube environment
from rubixRL.envs import RubiksCubeEnv


def make_env():
    """Create a wrapped environment."""
    env = gym.make("RubiksCube-v0", n=3, max_steps=100, reward_type="dense")
    return env


def evaluate_model(model_path: str, vec_normalize_path: str, n_episodes: int = 10):
    """Evaluate a trained model."""
    # Create environment
    env = DummyVecEnv([make_env])

    # Load the saved VecNormalize object
    env = VecNormalize.load(vec_normalize_path)
    env.training = False  # Do not update the normalization statistics
    env.norm_reward = False  # Don't normalize rewards for evaluation

    # Load the trained model
    model = PPO.load(model_path, env=env)

    # Run evaluation episodes
    episode_rewards = []
    episode_steps = []
    solved_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            if info[0].get("is_solved", False):
                solved_count += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

    # Print evaluation statistics
    print("\nEvaluation Statistics:")
    print(f"Number of Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f}")
    print(f"Success Rate: {(solved_count / n_episodes) * 100:.1f}%")

    env.close()
    return episode_rewards, episode_steps


def main():
    # Update these paths to point to your trained model and normalization
    model_dir = "models/sb3_ppo_latest"  # Update this to your model directory
    model_path = os.path.join(model_dir, "best_model/best_model")
    vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_normalize_path):
        print(f"Model or normalization files not found in {model_dir}")
        return

    print("Starting evaluation...")
    episode_rewards, episode_steps = evaluate_model(
        model_path=model_path, vec_normalize_path=vec_normalize_path, n_episodes=10
    )


if __name__ == "__main__":
    main()
