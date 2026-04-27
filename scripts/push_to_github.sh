#!/usr/bin/env bash
set -euo pipefail
GITHUB_USER="YousefZahran1"
REPO_NAME="fake-news-mlops"
DESCRIPTION="Productionized fake-news classifier with MLflow tracking, FastAPI serving, Postgres logging, drift detection, and CI."

if [ ! -d ".git" ]; then git init -q; git checkout -b main; fi
git config user.email "youssefzahran.y@gmail.com"
git config user.name  "Youssef Ibrahim"

git add README.md LICENSE .gitignore .env.example requirements.txt
git commit -q -m "Initial scaffold: README, LICENSE, deps" || true

git add src/__init__.py src/train/
git commit -q -m "Add training pipeline with MLflow tracking + registry" || true

git add src/serve/
git commit -q -m "Add FastAPI inference service with Postgres logging" || true

git add src/monitor/
git commit -q -m "Add drift detector (TF-IDF distance baseline)" || true

git add tests/
git commit -q -m "Add tests for drift detector and API schema" || true

git add deploy/ docs/ .github/workflows/
git commit -q -m "Add Docker, docker-compose, init.sql, CI, docs" || true

if command -v gh >/dev/null 2>&1; then
  gh repo create "$GITHUB_USER/$REPO_NAME" --public --description "$DESCRIPTION" --source=. --remote=origin --push
else
  echo "Add remote and push:"
  echo "  git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
  echo "  git push -u origin main"
fi
