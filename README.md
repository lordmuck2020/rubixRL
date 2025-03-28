# RubixRL

A Gymnasium-based Reinforcement Learning Environment for Rubik's Cube.

## Features

- Customizable n×n×n Rubik's Cube environment (3×3×3 by default)
- Full compatibility with Gymnasium API
- Support for cube scrambling
- Visualisation of the cube state in text format
- Flexible reward structure

## Installation

```bash
# Clone the repository
git clone https://github.com/username/rubixRL.git
cd rubixRL

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
import gymnasium as gym
import rubixRL

# Create the environment
env = gym.make('RubiksCube-v0')

# Reset the environment
state, info = env.reset(options={'scramble': True, 'scramble_moves': 10})

# Take a step
action = env.action_space.sample()  # Random action
next_state, reward, terminated, truncated, info = env.step(action)

# Render the current state
print(env.render())
```

### Custom Cube Size

```python
# Create a 2×2×2 Rubik's Cube environment
env = gym.make('RubiksCube-v0', n=2)

# Create a 4×4×4 Rubik's Cube environment
env = gym.make('RubiksCube-v0', n=4)
```

## State Representation

The state of the Rubik's Cube is represented as a 6×(n²) array, where:
- Each row corresponds to a face (UP, DOWN, FRONT, BACK, RIGHT, LEFT)
- Each element represents the colour of a specific cubelet (0-5)

For a 3×3×3 cube, the state shape is (6, 9).

## Action Space

For simplicity, the action space consists of 12 possible moves:
- Actions 0-5: Clockwise turns of faces 0-5
- Actions 6-11: Counter-clockwise turns of faces 0-5

## Integration with RL Libraries

### Stable Baselines3

```python
from stable_baselines3 import PPO
import gymnasium as gym
import rubixRL

# Create the environment
env = gym.make('RubiksCube-v0')

# Create the RL agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_rubiks_cube")

# Load the model
model = PPO.load("ppo_rubiks_cube")

# Test the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(env.render())
    
    if terminated or truncated:
        obs, info = env.reset()
```

### RLlib

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import rubixRL

ray.init()

config = (
    PPOConfig()
    .environment("RubiksCube-v0")
    .rollouts(num_rollout_workers=4)
    .framework("torch")
    .training(
        model={"fcnet_hiddens": [256, 256]},
        lr=5e-5,
    )
    .evaluation(evaluation_num_episodes=10)
)

# Train the agent
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 100},
    checkpoint_freq=10,
)

ray.shutdown()
```

## Limitations

- The current implementation simplifies the rotation mechanics and only fully implements the UP face rotation for demonstration purposes.
- For a complete implementation, all 6 face rotations need to be properly implemented.
- The current state representation might not be optimal for learning efficiency.

## Future Improvements

- Complete implementation of all face rotations
- Alternative state representations (one-hot encoding, corner/edge orientations)
- Dense reward functions based on distance metrics
- GPU-accelerated state transitions
- 3D visualisation options 