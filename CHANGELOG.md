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

### Planned
- Implement base CartPole environment wrapper
- Add DQN (Deep Q-Network) agent
- Create training and evaluation scripts
- Add visualization tools for training progress
- Implement logging and metrics tracking

## [0.1.0] - 2025-06-15

### Added
- Project initialization based on Gymnasium's CartPole environment
- Documentation framework with README and CHANGELOG

[Unreleased]: https://github.com/yourusername/rl_with_nima/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/rl_with_nima/releases/tag/v0.1.0