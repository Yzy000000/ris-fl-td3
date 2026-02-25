"""
TD3 implementation adapted for RLFLEnv.

Contains:
- ReplayBuffer
- Actor (2 layers 64)
- Critic (2 critics, each 2 layers 512)
- TD3Agent class with select_action and update

This file expects `torch` to be installed.
"""
import copy
from collections import deque
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 10000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.storage = []

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, float(done))
        if self.size < self.max_size:
            self.storage.append(data)
            self.size += 1
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int):
        ind = np.random.randint(0, self.size, size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in ind:
            s, a, r, ns, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(ns, copy=False))
            dones.append(np.array(d, copy=False))
        return (
            torch.from_numpy(np.vstack(states)).float(),
            torch.from_numpy(np.vstack(actions)).float(),
            torch.from_numpy(np.vstack(rewards)).float(),
            torch.from_numpy(np.vstack(next_states)).float(),
            torch.from_numpy(np.vstack(dones)).float(),
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        in_dim = state_dim + action_dim
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3Agent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 5e-4,
                 gamma: float = 0.99,
                 tau: float = 0.001,
                 buffer_size: int = 10000,
                 batch_size: int = 128,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 exploration_noise: float = 0.1,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr_critic)

        self.replay = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.total_it = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_t = torch.from_numpy(state.reshape(1, -1)).float().to(self.device)
        action = self.actor(state_t).cpu().data.numpy().flatten()
        if add_noise:
            action = action + np.random.normal(0, self.exploration_noise, size=action.shape)
        return action.clip(-1.0, 1.0)

    def update(self):
        if self.replay.size < self.batch_size:
            return
        self.total_it += 1

        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            # target policy smoothing
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-1.0, 1.0)
            q1_next = self.critic1_target(s2, a2)
            q2_next = self.critic2_target(s2, a2)
            q_next = torch.min(q1_next, q2_next)
            target_q = r + (1.0 - d) * self.gamma * q_next

        # current Q estimates
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)
        loss_critic = loss_q1 + loss_q2

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # actor loss
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(), path_prefix + '_actor.pth')
        torch.save(self.critic1.state_dict(), path_prefix + '_critic1.pth')
        torch.save(self.critic2.state_dict(), path_prefix + '_critic2.pth')

    def load(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(path_prefix + '_actor.pth', map_location=self.device))
        self.critic1.load_state_dict(torch.load(path_prefix + '_critic1.pth', map_location=self.device))
        self.critic2.load_state_dict(torch.load(path_prefix + '_critic2.pth', map_location=self.device))
