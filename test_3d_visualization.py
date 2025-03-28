import numpy as np
from rubixRL.envs.rubiks_cube_env import RubiksCubeEnv


def test_3d_visualization():
    print("Testing 3D Visualization")
    print("-" * 50)

    # Create a 3x3x3 Rubik's Cube environment
    env = RubiksCubeEnv(n=3)

    # Reset with scrambling
    state, info = env.reset(options={"scramble": True, "scramble_moves": 10})

    print("Initial state (ASCII):")
    print(env.render(mode="ansi"))
    print("\nOpening 3D visualization...")
    print("Controls:")
    print("- Arrow keys: Rotate view")
    print("- ESC: Close visualization")

    # Show 3D visualization
    env.render(mode="3d")

    # Take some random actions and show the cube after each
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"\nAction {i+1}: {action}")
        print("ASCII view:")
        print(env.render(mode="ansi"))
        print("\nOpening 3D visualization...")
        env.render(mode="3d")

        if terminated or truncated:
            break


if __name__ == "__main__":
    test_3d_visualization()
