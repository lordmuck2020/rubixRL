import numpy as np
from rubixRL.envs.rubiks_cube_env import RubiksCubeEnv


def test_extended_actions():
    print("Testing Extended Action Space")
    print("-" * 50)

    # Test 3x3x3 with extended actions
    env = RubiksCubeEnv(n=3, extended_action_space=True)
    print("\n1. Testing 3x3x3 with extended actions:")
    print(f"Action space size: {env.action_space.n}")  # Should be 18

    # Test 180° rotation
    state, _ = env.reset()
    print("\nInitial state:")
    print(env.render())

    # Apply a 180° rotation (U2)
    action = 12  # First 180° rotation (UP face)
    next_state, reward, terminated, truncated, info = env.step(action)
    print("\nAfter U2 move (180° rotation of UP face):")
    print(env.render())

    # Test larger cube with inner layer moves
    env = RubiksCubeEnv(n=4, extended_action_space=True)
    print("\n2. Testing 4x4x4 with extended actions:")
    print(f"Action space size: {env.action_space.n}")

    state, _ = env.reset()
    print("\nInitial state:")
    print(env.render())

    # Test inner layer move
    action = 18  # First inner layer move
    next_state, reward, terminated, truncated, info = env.step(action)
    print("\nAfter inner layer move:")
    print(env.render())


if __name__ == "__main__":
    test_extended_actions()
