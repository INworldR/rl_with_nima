# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- README.md with project overview and structure
- CHANGELOG.md for tracking version history
- Basic project directory structure planning
- README_smartpull.md documentation for git smartpull alias
- setup-git-aliases.sh script for easy git alias installation
- src/ directory with initial Python package structure
- src/__init__.py with package metadata and imports
- src/main.py with CartPoleExperiment class and CLI interface
  - Basic experiment framework with logging
  - Random agent baseline implementation
  - Command-line arguments for seed, rendering, and episode count
  - Episode statistics tracking and reporting

### Changed
- Updated README.md with complete project structure and usage instructions
- Enhanced project description to reflect learned dynamics implementation

### Implemented
- CartPole_Learned: Custom environment with neural network dynamics
- CartPole_Learned2: Extended version inheriting from Gymnasium's CartPoleEnv
- DQN agent implementation using Ray RLlib with advanced features:
  - Double Q-learning
  - Dueling network architecture
  - Experience replay buffer
  - Target network updates
- Random agent baseline for performance comparison
- Visualization utilities for environment rendering
- Training progress tracking with tqdm
- Comprehensive dependency list in requirements.txt

### Merged
- Integrated blue-team branch with data collection features:
  - Enhanced random_agent.py with dual functionality
  - Added collect_interaction_history() function for gathering state-action-next state pairs
  - Saves interaction data to pickle files for dynamics model training
  - Maintains original demo functionality with run_random_agent_demo()
  - Creates data directory automatically when saving histories

### Dependencies Added
- gymnasium>=0.28.1
- keras~=2.15.0
- matplotlib>=3.6.3
- numpy>=1.21.0
- pandas~=2.2.2
- pygame>=2.5.0
- pyglet==2.0.14
- pylint==3.1.0
- ray[default]==2.9.3
- ray[rllib]>=2.9.3
- tensorflow>=2.15.0
- torch>=2.7.1
- tqdm==4.65.0

## [0.1.0] - 2025-06-15

### Added
- Project initialization based on Gymnasium's CartPole environment
- Documentation framework with README and CHANGELOG

[Unreleased]: https://github.com/yourusername/rl_with_nima/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/rl_with_nima/releases/tag/v0.1.0