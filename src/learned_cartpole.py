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
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

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


class CartPole_Learned(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    
    def __init__(self, dynamics_path: Optional[str] = None, hidden_dim: int = 64, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.length = 0.5  # pole length
        self.max_steps =-200  # Maximum number of steps before termination
        self.steps = 0  # Step counter
        
        # Rendering parameters
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.surf = None
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        
        # Match the observation space with CartPole-v1
        self.observation_space = spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]),
            high=np.array([self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]),
            dtype=np.float32
        )

        # Set default model path if none provided
        if dynamics_path is None:
            dynamics_path = str(Path("model/team_blue_model.pkl"))
            
        # Initialize dynamics network
        if os.path.exists(dynamics_path):
            try:
                with open(dynamics_path, 'rb') as f:
                    self.dynamics = pickle.load(f)
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
        
        # Initialize state
        self.state = None
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Initialize state with small random values within bounds
        self.state = np.array([
            np.random.uniform(low=-0.05, high=0.05),  # x
            np.random.uniform(low=-0.05, high=0.05),  # x_dot
            np.random.uniform(low=-0.05, high=0.05),  # theta
            np.random.uniform(low=-0.05, high=0.05)   # theta_dot
        ], dtype=np.float32)
        self.steps = 0  # Reset step counter
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self.state, {}
        
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
            
            # Increment step counter
            self.steps += 1
            
            # Check if episode is done
            x, x_dot, theta, theta_dot = self.state
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
                or self.steps >= self.max_steps  # Terminate after max_steps
            )
            
            # Simple reward: 1 for each step
            reward = 1.0
            
            if self.render_mode == "human":
                self.render()
                
            return self.state, reward, done, False, {}
            
    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        if self.state is None:
            return None
        x = self.state
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

