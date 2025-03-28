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

## State Representations

The environment supports three different state representations:

1. **Flat Representation** (`state_type="flat"`):
   - The original 6×(n²) array representation
   - Each row corresponds to a face (UP, DOWN, FRONT, BACK, RIGHT, LEFT)
   - Each element represents the colour of a specific cubelet (0-5)
   - Shape: (6, n*n)
   - Most intuitive but may not be optimal for learning

2. **One-hot Encoding** (`state_type="onehot"`):
   - Each cubelet is represented by a 6-dimensional one-hot vector
   - The vector indicates which colour the cubelet has
   - Shape: (6, n*n, 6)
   - May help neural networks learn better by providing explicit colour information
   - More memory-intensive but potentially better for learning

3. **Corner/Edge Representation** (`state_type="corner_edge"`):
   - Specifically designed for 3×3×3 cubes
   - Represents the cube state in terms of corner and edge orientations
   - Shape: (8*3 + 12*2 + 6,) - one-hot encoded vector
   - Components:
     - 8 corners with 3 possible orientations each
     - 12 edges with 2 possible orientations each
     - 6 face centers with 1 orientation each
   - More compact and potentially better for learning cube-solving strategies
   - Only available for 3×3×3 cubes

### Example Usage with Different State Representations

```python
import gymnasium as gym
import rubixRL

# Create environment with flat state representation
env_flat = gym.make('RubiksCube-v0', state_type='flat')

# Create environment with one-hot state representation
env_onehot = gym.make('RubiksCube-v0', state_type='onehot')

# Create environment with corner/edge state representation
env_corner_edge = gym.make('RubiksCube-v0', state_type='corner_edge')

# Use the environments as before
state, info = env_onehot.reset(options={'scramble': True})
next_state, reward, terminated, truncated, info = env_onehot.step(action)
```

### Choosing a State Representation

1. **Flat Representation**:
   - Best for general use and debugging
   - Most intuitive to understand
   - Works with any cube size
   - Good for visualization

2. **One-hot Encoding**:
   - Better for neural network learning
   - Provides explicit colour information
   - Works with any cube size
   - More memory-intensive

3. **Corner/Edge Representation**:
   - Most efficient for 3×3×3 cubes
   - Better captures the structural properties of the cube
   - Limited to 3×3×3 cubes
   - May help with learning cube-solving strategies

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

## Reward Functions

The environment supports three types of reward functions:

1. **Sparse Reward** (`reward_type="sparse"`):
   - Returns 1.0 when the cube is solved
   - Returns 0.0 for all other states
   - This is the most challenging reward structure for learning

2. **Dense Reward** (`reward_type="dense"`):
   - Returns a reward between 0 and 1 based on the number of correctly placed cubelets
   - Calculated as: `correct_cubelets / total_cubelets`
   - Provides immediate feedback on progress
   - Helps guide the agent towards the solution

3. **Distance-based Reward** (`reward_type="distance"`):
   - Returns a reward between 0 and 1 based on the number of unsolved faces
   - Calculated as: `1.0 - (unsolved_faces / 6.0)`
   - Provides a different perspective on progress
   - May help the agent focus on solving one face at a time

### Example Usage with Different Reward Types

```python
import gymnasium as gym
import rubixRL

# Create environment with sparse reward
env_sparse = gym.make('RubiksCube-v0', reward_type='sparse')

# Create environment with dense reward
env_dense = gym.make('RubiksCube-v0', reward_type='dense')

# Create environment with distance-based reward
env_distance = gym.make('RubiksCube-v0', reward_type='distance')

# Use the environments as before
state, info = env_dense.reset(options={'scramble': True})
next_state, reward, terminated, truncated, info = env_dense.step(action)
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

---

- [x] More Efficient State Representation:
- The current representation uses a 6×(n²) array, which might not be the most efficient for learning.
- Alternative representations like one-hot encoding or directly encoding corner/edge orientations could be implemented.
- [x] Dense Reward Functions:
- The current sparse reward might make learning challenging.
- Adding dense rewards based on metrics like the number of correctly placed cubelets or distance to the solved state could help.
- [x] Extended Action Space:
- For cubes larger than 3×3×3, inner layer moves could be added to the action space.
- Double turns (180-degree rotations) could be added as separate actions to potentially speed up learning.
- [x] 3D Visualization:
- A 3D visualization option could make it easier to understand the cube state and debug the environment.