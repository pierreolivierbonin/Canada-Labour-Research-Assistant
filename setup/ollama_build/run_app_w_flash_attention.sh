#!/usr/bin/env bash
source ./.venv/scripts/activate
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
ollama pull "llama3.2:latest" &&
ollama serve &
streamlit run ./chatbot_app.py