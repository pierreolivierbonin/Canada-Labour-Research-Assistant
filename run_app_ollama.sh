#!/bin/bash

# Activate virtual environment (cross-platform compatible)
if [ -f ./.venv/bin/activate ]; then
    # Unix/Linux/WSL
    source ./.venv/bin/activate
elif [ -f ./.venv/Scripts/activate ]; then
    # Windows
    source ./.venv/Scripts/activate
else
    echo "Virtual environment not found. Please create one first."
    exit 1
fi

# Create .models directory if it doesn't exist
mkdir -p .models

if [ ! -f .models/summac_conv_vitc_sent_perc_e.bin ]; then
    curl -L https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin --output .models/summac_conv_vitc_sent_perc_e.bin
else
    printf "\nSummaC model file found"
fi

# Check and download NLTK punkt tokenizer
printf "\nChecking NLTK punkt tokenizer..."
if python -c "import nltk; nltk.find('tokenizers/punkt_tab.zip')" 2>/dev/null; then
    printf "\nNLTK tokenizer 'punkt' found"
else
    printf "\nNLTK tokenizer 'punkt' not found, downloading..."
    python -c "import nltk; nltk.download('punkt_tab')"
    printf "\nNLTK tokenizer download completed"
fi 

printf "\n\nHello! Please select the mode in which you want to launch this application: local, remote, evaluation_local, evaluation_remote, profiling_local or profiling_remote\n"

read user_input

# export OLLAMA_FLASH_ATTENTION=1       # uncomment if you wanna use flash attention
# export OLLAMA_KV_CACHE_TYPE=q8_0      # uncomment if you wanna use a quantized version of the model


# Check if Ollama is already running, if not start it
if command -v pgrep &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama server..."
        ollama serve &
        sleep 3
    else
        echo "Ollama server is already running"
    fi
fi

ollama pull "gemma3n:latest" &&
streamlit run ./chatbot_app.py $user_input
