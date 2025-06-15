# Git Smartpull Guide

## What is smartpull?

`smartpull` is a Git alias that safely updates your local repository by temporarily storing your uncommitted changes, pulling the latest updates with rebase, and then restoring your work. It's designed to prevent merge conflicts and maintain a clean Git history.

## How it works

The `smartpull` command executes the following sequence:

```bash
git stash push -m "temp" && git pull --rebase && git stash pop || true
```

### Step-by-step breakdown:

1. **`git stash push -m "temp"`**  
   Saves all your uncommitted changes to a temporary stash with the message "temp"

2. **`git pull --rebase`**  
   Fetches the latest changes from the remote repository and rebases your local commits on top of them (instead of creating a merge commit)

3. **`git stash pop`**  
   Restores your previously stashed changes back to your working directory

4. **`|| true`**  
   Ensures the command completes successfully even if there are conflicts during stash pop

## Installation

### Method 1: Using Git Config Command (Recommended)

Run this command in your terminal:

```bash
git config --global alias.smartpull '!git stash push -m "temp" && git pull --rebase && git stash pop || true'
```

### Method 2: Using the Setup Script

We've provided a setup script in this repository:

```bash
# Make the script executable (only needed once)
chmod +x setup-git-aliases.sh

# Run the setup script
./setup-git-aliases.sh
```

### Method 3: Manual Configuration

Add the following to your `~/.gitconfig` file:

```ini
[alias]
    smartpull = !git stash push -m "temp" && git pull --rebase && git stash pop || true
```

## Usage

Once installed, you can use smartpull like any other git command:

```bash
git smartpull
```

## Benefits

- **Prevents conflicts**: Automatically handles uncommitted changes that would otherwise block a pull
- **Cleaner history**: Uses rebase instead of merge to avoid unnecessary merge commits
- **Safer workflow**: Reduces the risk of losing work during updates
- **Time-saving**: Combines multiple commands into one simple alias

## When to use smartpull

Use `smartpull` when:
- You have local changes that aren't ready to commit yet
- You need to update your branch with the latest changes from remote
- You want to avoid creating merge commits for simple updates
- You're working on a feature branch and need to stay up-to-date with main/master

## Potential issues and solutions

### Stash conflicts
If you encounter conflicts when the stash is being popped, Git will notify you. You'll need to:
1. Resolve the conflicts manually
2. Run `git stash drop` to remove the temporary stash

### Rebase conflicts
If there are conflicts during the rebase:
1. Resolve the conflicts in the affected files
2. Run `git add <resolved-files>`
3. Continue with `git rebase --continue`

### Verifying installation

To check if smartpull is installed correctly:

```bash
git config --get alias.smartpull
```

This should output:
```
!git stash push -m "temp" && git pull --rebase && git stash pop || true
```

## Alternative aliases

You might also find these related aliases useful:

```bash
# Quick status
git config --global alias.st 'status -s'

# Pretty log
git config --global alias.lg 'log --oneline --graph --decorate'

# Amend last commit
git config --global alias.amend 'commit --amend --no-edit'
```

## Contributing

If you have suggestions for improving this workflow or the smartpull alias, please feel free to open an issue or submit a pull request!