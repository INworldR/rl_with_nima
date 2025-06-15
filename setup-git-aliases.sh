#!/bin/bash
# Setup useful git aliases for the project

echo "Setting up git aliases..."

# Add smartpull alias
git config --global alias.smartpull '!git stash push -m "temp" && git pull --rebase && git stash pop || true'

echo "✓ Added 'git smartpull' alias"
echo ""
echo "You can now use:"
echo "  git smartpull - Safely pull changes by stashing, pulling with rebase, and restoring your work"
echo ""
echo "To verify the alias was added, run: git config --get alias.smartpull"