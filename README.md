# Reinforcement Learning with CartPole

A reinforcement learning project based on the classic CartPole environment from OpenAI Gymnasium. This project implements and experiments with various RL algorithms to solve the pole balancing task.

## Overview

The CartPole environment consists of a pole attached by an un-actuated joint to a cart, which moves along a frictionless track. The goal is to prevent the pole from falling over by applying forces to move the cart left or right.

## Environment Details

### Action Space
- **Type**: Discrete(2)
- **Actions**:
  - 0: Push cart to the left
  - 1: Push cart to the right

### Observation Space
- **Type**: Box(4)
- **Observations**:
  1. Cart Position: [-4.8, 4.8]
  2. Cart Velocity: [-Inf, Inf]
  3. Pole Angle: [-0.418 rad, 0.418 rad] (~24°)
  4. Pole Angular Velocity: [-Inf, Inf]

### Rewards
- **Default**: +1 for every timestep the pole remains upright
- **Episode Termination**:
  - Pole angle exceeds ±12°
  - Cart position goes beyond ±2.4
  - Episode length reaches 500 steps (considered solved)

## Project Structure

```
rl_with_nima/
├── README.md           # This file
├── CHANGELOG.md        # Version history
├── requirements.txt    # Python dependencies
├── src/               # Source code
│   ├── agents/        # RL agents implementations
│   ├── environments/  # Custom environment wrappers
│   └── utils/         # Utility functions
├── experiments/       # Experiment scripts and configs
├── models/           # Saved models
└── results/          # Training logs and visualizations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl_with_nima.git
cd rl_with_nima

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Example usage will be added as the project develops
```

## Algorithms

This project will explore various RL algorithms:
- [ ] Deep Q-Network (DQN)
- [ ] Policy Gradient Methods
- [ ] Actor-Critic Methods
- [ ] PPO (Proximal Policy Optimization)
- [ ] More to be added...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the CartPole environment from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- Inspired by classic control theory and reinforcement learning research