#!/usr/bin/env bash
source ./.venv/scripts/activate
if [ ! -f .models/summac_conv_vitc_sent_perc_e.bin ]; then
    curl -L https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin --output .models/summac_conv_vitc_sent_perc_e.bin
else
    printf "\nSummaC model file found"
fi

python -c "import nltk; nltk.find('tokenizers/punkt_tab.zip')" & printf "\nNLTK tokenizer 'punkt' found" || python -c "import nltk; nltk.download('punkt_tab')" 

printf "\n\nHello! Please select the mode in which you want to launch this application: local, remote, evaluation_local, evaluation_remote, profiling_local or profiling_remote\n"

read user_input

ollama pull "llama3.2:latest" &&
ollama serve # stops any ollama process already running before running it here (https://github.com/ollama/ollama/issues/3575)
streamlit run ./chatbot_app.py $user_input