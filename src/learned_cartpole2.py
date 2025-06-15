import math
from typing import Union, Optional, Tuple
import pickle
import os
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

model_path = "data/random_agent_history.pkl"
# Set up logger
logger = logging.getLogger(__name__)

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


class CartPole_Learned2(CartPoleEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    
    def __init__(self, dynamics_path: Optional[str] = None, hidden_dim: int = 64, render_mode: Optional[str] = None):
        super().__init__(render_mode=render_mode)
        
        # Set default model path if none provided
        if dynamics_path is None:
            dynamics_path = str(Path(model_path))
            
        # Initialize dynamics network
        if os.path.exists(dynamics_path):
            try:
                with open(dynamics_path, 'rb') as f:
                    model_data = pickle.load(f)
                    # Create new network and load state dict
                    self.dynamics = DynamicsNetwork(
                        state_dim=4,
                        action_dim=1,
                        hidden_dim=hidden_dim
                    )
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        self.dynamics.load_state_dict(model_data['state_dict'])
                    else:
                        # If it's a DataFrame or other format, we need to convert it
                        logger.warning("Model format not recognized, using untrained network")
                logger.info(f"Successfully loaded dynamics model from {dynamics_path}")
            except Exception as e:
                logger.warning(f"Failed to load dynamics model from {dynamics_path}: {e}")
                self.dynamics = DynamicsNetwork(
                    state_dim=4,
                    action_dim=1,
                    hidden_dim=hidden_dim
                )
        else:
            logger.warning(f"Dynamics model not found at {dynamics_path}, using untrained network")
            self.dynamics = DynamicsNetwork(
                state_dim=4,
                action_dim=1,
                hidden_dim=hidden_dim
            )
        
        self.dynamics.eval()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        with torch.no_grad():
            # Normalize state for network input
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
            action_tensor = torch.FloatTensor([[action]])
            
            # Predict next state using dynamics network
            next_state = self.dynamics(state_tensor, action_tensor)
            self.state = next_state.squeeze().numpy()
            
            # Ensure state is within bounds
            self.state = np.clip(
                self.state,
                self.observation_space.low,
                self.observation_space.high
            )
            
            # Check if episode is done
            x, x_dot, theta, theta_dot = self.state
            terminated = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
            
            # Simple reward: 1 for each step
            reward = 1.0
            
            if self.render_mode == "human":
                self.render()
            
            return self.state, reward, terminated, False, {} 