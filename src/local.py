from ollama import chat

def get_ollama_answer_local(chat_model, messages, hyperparams):
    answer = chat(model=chat_model,
                  messages=messages,
                  options=hyperparams)
    
    return answer["message"]["content"]

def get_ollama_answer_local_stream(chat_model, messages, hyperparams):
    stream = chat(model=chat_model,
                  messages=messages,
                  options=hyperparams,
                  stream=True)

    for chunk in stream:
        content = chunk.get('message', {}).get('content')
        if content:
            yield content