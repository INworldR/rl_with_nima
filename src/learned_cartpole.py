import math
from typing import Union, Optional, Tuple
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(DynamicsNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CartPole_Learned(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, dynamics_path: Optional[str] = None, hidden_dim: int = 64):
        super().__init__()
        
        # Environment parameters
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]),
            high=np.array([self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]),
            dtype=np.float32
        )

        # Initialize dynamics network
        if dynamics_path and os.path.exists(dynamics_path):
            with open(dynamics_path, 'rb') as f:
                self.dynamics = pickle.load(f)
        else:
            self.dynamics = DynamicsNetwork(
                state_dim=4,
                action_dim=1,
                hidden_dim=hidden_dim
            )
        
        self.dynamics.eval()
        
        # Initialize state
        self.state = None
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Initialize with random state within reasonable bounds
        self.state = np.array([
            np.random.uniform(-0.5, 0.5),  # cart position
            np.random.uniform(-0.5, 0.5),  # cart velocity
            np.random.uniform(-0.1, 0.1),  # pole angle
            np.random.uniform(-0.1, 0.1)   # pole angular velocity
        ], dtype=np.float32)
        return self.state, {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        with torch.no_grad():
            # Normalize state for network input
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
            action_tensor = torch.FloatTensor([[action]])
            
            # Predict next state using dynamics network
            next_state = self.dynamics(state_tensor, action_tensor)
            self.state = next_state.squeeze().numpy()
            
            # Check if episode is done
            x, x_dot, theta, theta_dot = self.state
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
            
            # Simple reward: 1 for each step
            reward = 1.0
            
            return self.state, reward, done, False, {}

