#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f data/bizcopilot.db ]; then
  echo "Generating synthetic data..."
  python3 data/generate_data.py
fi

echo "Starting BizCopilot at http://localhost:8000"
uvicorn app.main:app --reload --port 8000
