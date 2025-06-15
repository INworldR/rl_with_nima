import math
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__():