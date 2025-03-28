"""
Example usage of the Rubik's Cube Gymnasium environment.
"""

import gymnasium as gym
import numpy as np
import rubixRL
import time


def test_state_type(env, state_type: str, num_steps: int = 3):
    """
    Test the environment with a specific state representation.

    Args:
        env: The Rubik's Cube environment.
        state_type (str): The type of state representation to use.
        num_steps (int): Number of steps to take.
    """
    print(f"\nTesting with {state_type} state representation:")
    print(f"Observation space: {env.observation_space}")

    state, info = env.reset(options={"scramble": True, "scramble_moves": 5})

    print(f"\nInitial state shape: {state.shape}")
    if state_type == "flat":
        print("Initial state:")
        print(env.render())
    print(f"Is solved: {info['is_solved']}")

    total_reward = 0
    for i in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {i+1}, Action: {action}")
        if state_type == "flat":
            print(env.render())
        print(f"State shape: {next_state.shape}")
        print(f"Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
        print(f"Is solved: {info['is_solved']}")

        if terminated:
            print("\nCube solved! Episode terminated.")
            break

        if truncated:
            print("\nMaximum number of steps reached. Episode truncated.")
            break

        time.sleep(0.5)


def main():
    """Main function to demonstrate the Rubik's Cube environment."""

    print("Creating a 3x3x3 Rubik's Cube environment...")

    # Test with flat state representation
    env_flat = gym.make("RubiksCube-v0", state_type="flat")
    test_state_type(env_flat, "flat")
    env_flat.close()

    # Test with one-hot state representation
    env_onehot = gym.make("RubiksCube-v0", state_type="onehot")
    test_state_type(env_onehot, "onehot")
    env_onehot.close()

    # Test with corner/edge state representation
    env_corner_edge = gym.make("RubiksCube-v0", state_type="corner_edge")
    test_state_type(env_corner_edge, "corner_edge")
    env_corner_edge.close()

    print("\nExample completed!")


if __name__ == "__main__":
    main()
