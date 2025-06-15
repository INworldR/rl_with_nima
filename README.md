# Reinforcement Learning with CartPole

A reinforcement learning project based on the classic CartPole environment from OpenAI Gymnasium. This project implements learned dynamics models and DQN agents to solve the pole balancing task using neural network-based environment simulation.

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
├── README.md                  # This file
├── README_smartpull.md        # Git workflow documentation
├── CHANGELOG.md               # Version history
├── requirements.txt           # Python dependencies
├── setup-git-aliases.sh       # Git configuration script
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main experiment runner
│   ├── random_agent.py       # Random agent with demo and data collection
│   ├── learned_agent.py      # DQN agent with RLlib
│   ├── learned_cartpole.py   # Learned dynamics CartPole v1
│   ├── learned_cartpole2.py  # Learned dynamics CartPole v2
│   └── plot_util.py          # Visualization utilities
├── data/                      # Collected interaction histories
├── model/                     # Trained dynamics models
└── mean_reward_vs_training_rounds_learned.png  # Training results
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

### 1. Run the Basic Experiment Framework
```bash
# Run with random actions (baseline)
python src/main.py --episodes 10

# Run with rendering enabled
python src/main.py --episodes 5 --render

# Run with specific seed for reproducibility
python src/main.py --episodes 10 --seed 123
```

### 2. Test Random Agent on Learned Environment
```bash
python src/random_agent.py
```

This will:
- Run a demo showing random agent performance
- Collect 100 episodes of interaction data
- Save state-action-next state pairs to `data/random_agent_history.pkl`

### 3. Train DQN Agent with RLlib
```bash
python src/learned_agent.py
```

This will:
- Train a DQN agent for 100 iterations
- Show progress with a progress bar
- Generate a plot of mean rewards over training rounds
- Visualize the trained agent's performance

## Components

### Environments
- **CartPole_Learned**: Custom CartPole environment with learned dynamics using neural networks
  - Default model path: `model/team_blue_model.pkl`
  - Supports direct pickle loading of trained models
- **CartPole_Learned2**: Extended version inheriting from Gymnasium's CartPoleEnv
  - Default model path: `data/random_agent_history.pkl`
  - Enhanced model loading with PyTorch state_dict format support
  - Automatic format detection (DataFrame vs PyTorch state_dict)

### Agents
- **Random Agent**: Baseline agent with two modes:
  - Demo mode: Shows performance on learned CartPole environment
  - Data collection mode: Gathers interaction histories for training dynamics models
- **DQN Agent**: Deep Q-Network implementation using Ray RLlib with:
  - Experience replay buffer
  - Target network updates
  - Double Q-learning
  - Dueling network architecture

### Key Features
- Learned dynamics models using PyTorch neural networks
- Flexible model loading supporting multiple formats (pickle, PyTorch state_dict)
- Integration with Ray RLlib for scalable RL training
- Data collection utilities for training dynamics models
- Visualization utilities for environment rendering
- Progress tracking with tqdm
- Configurable experiment parameters via CLI
- Automatic directory creation for data storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the CartPole environment from [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- Inspired by classic control theory and reinforcement learning research