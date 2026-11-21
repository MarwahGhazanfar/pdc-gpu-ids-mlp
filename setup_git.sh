#!/usr/bin/env bash
# Usage: ./setup_git.sh https://github.com/<your-username>/pdc-gpu-ids-mlp.git
# Create an EMPTY PUBLIC repo on GitHub first (no README/license), then run this once.
set -e
REMOTE="$1"
[ -z "$REMOTE" ] && { echo "Provide the GitHub repo URL"; exit 1; }

git init -b main
git add .
git commit -m "repo: initial README, layout and branching guide"
git tag -a v0.0-scope -m "Phase 0 scope pack and repository skeleton"
git branch develop
git remote add origin "$REMOTE"
git push -u origin main
git push -u origin develop
git push origin v0.0-scope
echo "Done. Now protect 'main' and add the instructor as collaborator (see SETUP_STEPS.md)."
