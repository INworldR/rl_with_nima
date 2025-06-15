"""Utility functions for visualization."""
import matplotlib.pyplot as plt
import numpy as np


def visualize_env(env, pause_sec=0.1):
    """Visualize the current state of the environment."""
    plt.clf()
    plt.imshow(env.render())
    plt.axis('off')
    plt.pause(pause_sec) 