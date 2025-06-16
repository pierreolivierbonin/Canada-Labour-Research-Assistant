#!/usr/bin/env bash
python -m venv .venv
source ./.venv/scripts/activate
pip install uv
uv pip install -e .