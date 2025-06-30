#!/usr/bin/env bash
python3 -m venv .venv
# Activate virtual environment (cross-platform compatible)
if [ -f ./.venv/bin/activate ]; then
    # Unix/Linux/WSL
    source ./.venv/bin/activate
elif [ -f ./.venv/Scripts/activate ]; then
    # Windows
    source ./.venv/Scripts/activate
fi
pip install uv
cd .setup_vllm
uv pip install -e .
cd ..