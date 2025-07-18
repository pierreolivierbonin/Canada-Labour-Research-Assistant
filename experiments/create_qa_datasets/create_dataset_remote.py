from datasets import load_dataset, Dataset
import time

import requests
import toml
import json
import re

def read_secrets(file_path):
    """Reads secrets from a TOML file."""
    with open(file_path, "r") as f:
        secrets = toml.load(f)
    return secrets

# Example Usage
file_path = "/home/mark/.secrets/secrets.toml"
secrets = read_secrets(file_path)

# Accessing secrets
authorization = secrets["authorization"]
api_url = secrets["api_url"]

HyperparametersAccuracyConfig = {
    "mirostat_tau":0,
    "seed":1837,
    "num_ctx": 16000, 
    "temperature": 0.0,
    "top_k":1,
    "top_p":0.1 # Top P is not used unless you set the Top P parameter value to something other than the default value of 1.
}

def get_remote_params(chat_model, messages, is_stream):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {authorization}"
    }

    data = {
        "model": chat_model,
        "messages": messages,
        "stream": is_stream,
        "temperature": HyperparametersAccuracyConfig.get("temperature", None),
        "top_p": HyperparametersAccuracyConfig.get("top_p", None),
        "max_tokens": HyperparametersAccuracyConfig.get("num_ctx", None)
    }

    return headers, data

def get_llm_answer_remote(chat_model, messages):
    headers, data = get_remote_params(chat_model, messages, False)

    # Make the request
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code != 404:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return None
    
def extract_json_from_response(response_content):
    try:
        matches = re.findall(r'\{.*\}|\{.*$', response_content, re.MULTILINE | re.DOTALL)
        
        if matches:
            modified_response = matches[0]

            missing_closing_braces = modified_response.count('{') - modified_response.count('}')
            if missing_closing_braces > 0:
                modified_response += '}' * missing_closing_braces

            # Remove trailing commas right before ] or }
            modified_response = re.sub(r',\s*([}\]])', r'\1', modified_response)

            json_obj = json.loads(modified_response)

            if not isinstance(json_obj, dict):
                print(f'Error. {response_content}')
                return None
            
            return json_obj
        else:
            print(f'Error: No potential JSON object found in the text. {response_content}')
            return None
    except json.JSONDecodeError as e:
        print(f'Error: The text does not contain a valid JSON object. {response_content}. Error : {str(e)}')
        return None

start_time = time.time()

# Create a dataset based on clc_dataset/data.csv, where the text column is the unannotated text
clc_dataset = load_dataset("csv", data_files="experiments/create_qa_datasets/clc_dataset/data.csv", split="train", streaming=False) #.select(range(10))

questions_answers = []

# Loop over all the sections of the CLC (except the first one)
for i in range(1, len(clc_dataset)):
    section_number = clc_dataset[i]['section_number']
    text = clc_dataset[i]['text']

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates questions based on sections of the Canada Labour Code (CLC)."},
        {"role": "user", "content": text},
        {"role": "user", "content": "Generate a question based on the given section of the Canada Labour Code (CLC). It should be possible to quotes passages from the section given to answer the question. The question should not refer to specific parts of the section, but rather to the general idea of the section. Then, give a short answer to the question. \nYour response should be fully in json, in the following format: {\"explanation\": \"[short explanation for why the question was chosen]\", \"question\": \"[question]\", \"answer\": \"[short answer]\"}"}
    ]
    
    # Ask the LLM to generate a question and answer based on the text
    json_response = get_llm_answer_remote("meta-llama/Llama-4-Scout-17B-16E-Instruct", messages)
    json_response = extract_json_from_response(json_response)
    question = json_response['question']
    answer = json_response['answer']

    questions_answers.append({
        "section_number": section_number,
        "question": question,
        "answer": answer
    })

    print(questions_answers)

# Save the questions and answers to a json file
with open("questions_answers.json", "w") as f:
    json.dump(questions_answers, f)