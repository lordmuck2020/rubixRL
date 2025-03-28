import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Dict, Any, List
import gymnasium as gym
from collections import deque
import random


class RubiksCubePPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(RubiksCubePPO, self).__init__()

        # Policy network (actor)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Value network (critic)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten the state if it's not already flattened
        if len(state.shape) > 2:
            state = state.reshape(state.size(0), -1)
        action_probs = self.policy(state)
        state_value = self.value(state)
        return action_probs, state_value


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 10,
        hidden_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.device = device

        # Get state and action dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            # For the Rubik's Cube, state shape is (6, 9) for 3x3x3
            state_dim = np.prod(env.observation_space.shape)
        else:
            state_dim = env.observation_space.n

        action_dim = env.action_space.n

        # Initialize policy network
        self.policy = RubiksCubePPO(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Initialize memory
        self.memory = deque(maxlen=10000)

    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.policy(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state_value

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
        dones: List[bool],
    ) -> torch.Tensor:
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, device=self.device)

    def update(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {}

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, rewards, next_states, dones = zip(*batch)

        # Convert to tensors efficiently
        states = np.array(states)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get next state value for GAE
        next_state = torch.FloatTensor(next_states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy(next_state)

        # Compute advantages once
        with torch.no_grad():
            _, values = self.policy(states)
            advantages = self.compute_gae(rewards, values, next_value, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0

        for _ in range(self.n_epochs):
            # Get current policy outputs
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # Compute ratio (create new tensor)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses (create new tensors)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )

            # Compute policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Compute value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)

            # Compute entropy loss
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # Accumulate losses for averaging
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()

            # Clear memory
            del loss, policy_loss, value_loss, entropy_loss, surr1, surr2, ratio
            torch.cuda.empty_cache()

        # Return average losses
        return {
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs,
            "entropy_loss": total_entropy_loss / self.n_epochs,
            "total_loss": total_loss / self.n_epochs,
        }

    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> List[float]:
        episode_rewards = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)

                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.memory.append((state, action, log_prob, reward, next_state, done))

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update policy
            metrics = self.update()

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
                if metrics:
                    print(f"Policy Loss: {metrics['policy_loss']:.4f}")
                    print(f"Value Loss: {metrics['value_loss']:.4f}")
                    print(f"Entropy Loss: {metrics['entropy_loss']:.4f}")

        return episode_rewards

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
