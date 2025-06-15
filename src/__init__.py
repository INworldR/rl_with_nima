#!/home/marc/miniconda3/envs/jupyter-ai/bin/python
"""
Reinforcement Learning with CartPole

A package for experimenting with reinforcement learning algorithms
on the classic CartPole control problem.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import agents
from . import environments
from . import utils

__all__ = ["agents", "environments", "utils"]