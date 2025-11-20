"""
DQN-MASK implementation used to reconfigure the shipboard power system.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .environment import ShipPowerEnv


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    epsilon_start: float = 0.9
    epsilon_final: float = 0.05
    epsilon_decay: int = 10_000
    target_update: int = 500
    hidden_size: int = 128
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        batch = dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
        )
        return batch

    def __len__(self):
        return self.capacity if self.full else self.idx


class DQNMaskAgent:
    def __init__(self, env: ShipPowerEnv, config: DQNConfig):
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)

        obs_dim = len(self.env._observation())
        action_dim = self.env.action_size
        self.policy_net = QNetwork(obs_dim, action_dim, self.cfg.hidden_size).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, self.cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optim = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim)

        self.steps_done = 0

    def _epsilon(self) -> float:
        eps = self.cfg.epsilon_final + (self.cfg.epsilon_start - self.cfg.epsilon_final) * (
            np.exp(-1.0 * self.steps_done / self.cfg.epsilon_decay)
        )
        return max(self.cfg.epsilon_final, eps)

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self.steps_done += 1
        if random.random() < self._epsilon():
            legal = np.where(mask > 0.5)[0]
            if len(legal) == 0:
                return self.env.action_size - 1  # fall back to no-op
            return int(random.choice(legal))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_t).squeeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
            masked_q = q_values + (mask_t - 1) * 1e9
            return int(torch.argmax(masked_q).item())

    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)

        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(1)[0]
            target = rewards + (1 - dones) * self.cfg.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optim.step()

        if self.steps_done % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def run_episode(self) -> Tuple[float, int]:
        obs = self.env.reset()
        total_reward = 0.0
        for _ in range(self.env.cfg.horizon):
            mask = self.env.legal_actions()
            action = self.select_action(obs, mask)
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            loss = self.train_step()
            if done:
                break
        return total_reward, len(self.buffer)


def train_agent(env: ShipPowerEnv, episodes: int, cfg: DQNConfig) -> Tuple[DQNMaskAgent, list]:
    agent = DQNMaskAgent(env, cfg)
    returns = []
    for ep in range(episodes):
        reward, _ = agent.run_episode()
        # Convert to native Python float to keep downstream JSON logging simple
        returns.append(float(reward))
    return agent, returns
