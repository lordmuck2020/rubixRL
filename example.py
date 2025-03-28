"""
Example usage of the Rubik's Cube Gymnasium environment.
"""

import gymnasium as gym
import numpy as np
import rubixRL
import time


def main():
    """Main function to demonstrate the Rubik's Cube environment."""

    print("Creating a 3x3x3 Rubik's Cube environment...")
    env = gym.make("RubiksCube-v0")

    print("\n1. Environment Information:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Reset the environment with scrambling
    print("\n2. Resetting the environment with scrambling...")
    state, info = env.reset(options={"scramble": True, "scramble_moves": 5})

    print("\n3. Initial state after scrambling:")
    print(env.render())
    print(f"Is solved: {info['is_solved']}")

    # Perform a sequence of random actions
    print("\n4. Taking a sequence of random actions...")

    total_reward = 0
    num_steps = 10

    for i in range(num_steps):
        # Sample a random action
        action = env.action_space.sample()

        # Apply the action
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"\nStep {i+1}, Action: {action}")
        print(env.render())
        print(f"Reward: {reward}, Total Reward: {total_reward}")
        print(f"Is solved: {info['is_solved']}")

        if terminated:
            print("\nCube solved! Episode terminated.")
            break

        if truncated:
            print("\nMaximum number of steps reached. Episode truncated.")
            break

        # Add a small delay for better visualization
        time.sleep(0.5)

    # Close the environment
    env.close()

    print("\n5. Creating a 2x2x2 Rubik's Cube environment...")
    env_2x2 = gym.make("RubiksCube-v0", n=2)
    state, info = env_2x2.reset(options={"scramble": True, "scramble_moves": 3})

    print("\n6. Initial state of 2x2x2 cube after scrambling:")
    print(env_2x2.render())

    env_2x2.close()

    print("\nExample completed!")


if __name__ == "__main__":
    main()
