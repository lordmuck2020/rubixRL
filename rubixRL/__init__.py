"""
RubixRL - A Gymnasium-based Reinforcement Learning Environment for Rubik's Cube
"""

from gymnasium.envs.registration import register

# Register the Rubik's Cube environment
register(
    id="RubiksCube-v0",
    entry_point="rubixRL.envs:RubiksCubeEnv",
    max_episode_steps=100,
)
