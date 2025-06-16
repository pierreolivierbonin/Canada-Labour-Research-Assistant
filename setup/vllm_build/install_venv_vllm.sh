#!/usr/bin/env bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
cd .setup_vllm
uv pip install -e .
cd ..