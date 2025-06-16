import logging
import streamlit as st
from oai import oai_compatible_request, oai_compatible_request_stream

# Remote-specific functions
def get_remote_params(chat_model, messages, hyperparams, is_stream):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['authorization']}"
    }

    data = {
        "model": chat_model,
        "messages": messages,
        "stream": is_stream,
        "temperature": hyperparams.get("temperature", None),
        "top_p": hyperparams.get("top_p", None),
        "max_tokens": hyperparams.get("num_ctx", None)
    }

    return headers, data

def get_llm_answer_remote(chat_model, messages, hyperparams):
    headers, data = get_remote_params(chat_model, messages, hyperparams, False)
    
    try:
        return oai_compatible_request(st.secrets['api_url'], headers, data)
    except Exception as e:
        if "404" in str(e):
            logging.error("Model not found")
        else:
            logging.error(f"Remote API error: {e}")
        return None

def get_llm_answer_remote_stream(chat_model, messages, hyperparams):
    headers, data = get_remote_params(chat_model, messages, hyperparams, True)
    
    try:
        for content in oai_compatible_request_stream(st.secrets['api_url'], headers, data):
            yield content
    except Exception as e:
        logging.error(f"Remote streaming API error: {e}")
        return None

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
    from config import OllamaRAGConfig

    for token in get_llm_answer_remote_stream(chat_model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                                              messages=[{   
                                                "role": "user",
                                                "content": "What are Ontario's primary economic drivers?"
                                                }],
                                              hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig):
        print(token, end="")