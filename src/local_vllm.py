from oai import oai_compatible_request, oai_compatible_request_stream

def get_vllm_params(chat_model, messages, hyperparams, is_stream, api_key=123):
    """Prepare vLLM-specific parameters"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": chat_model,
        "messages": messages,
        "stream": is_stream,
        **hyperparams
    }
    
    return headers, data

def get_vllm_api_url(api_url):
    return f"{api_url}/chat/completions"

def get_vllm_answer(chat_model, messages, hyperparams, api_key=123, api_url="http://localhost:8000/v1"):
    headers, data = get_vllm_params(chat_model, messages, hyperparams, False, api_key)
    return oai_compatible_request(get_vllm_api_url(api_url), headers, data)

def get_vllm_answer_stream(chat_model, messages, hyperparams, api_key=123, api_url="http://localhost:8000/v1"):
    headers, data = get_vllm_params(chat_model, messages, hyperparams, True, api_key)
    return oai_compatible_request_stream(get_vllm_api_url(api_url), headers, data)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path

    from config import vLLMRAGConfig

    model = "meta-llama/Llama-3.2-3B-Instruct"

    message = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "What are Canada's primary economic drivers?"
    }]

    answer = get_vllm_answer(chat_model=model, messages=message, hyperparams=vLLMRAGConfig.HyperparametersAccuracyConfig)
    print(answer)

    for word in get_vllm_answer_stream(chat_model=model, messages=message, hyperparams=vLLMRAGConfig.HyperparametersAccuracyConfig):
        print(word, end="")