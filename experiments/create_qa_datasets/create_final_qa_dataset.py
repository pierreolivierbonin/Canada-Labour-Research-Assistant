import json
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Add the parent directory to sys.path

from config import OllamaRAGConfig, ChatbotInterfaceConfig
from tools import retrieve_database_local
import time

def create_final_qa_dataset(model_name):
    engine = "ollama"
    
    # Read questions_answers.json
    with open("experiments/create_qa_datasets/new_questions_answers.json", "r", encoding="utf-8") as f:
        questions_answers = json.load(f)

    json_results = []
    total_start_time = time.time()
    filename = f"new_{model_name.replace(':', '_')}.json"
    
    for x, question_obj in enumerate(questions_answers):
        question = question_obj["question"]
        section_number = question_obj["section_number"]
        start_time = time.time()

        print(f"Question {x+1}/{len(questions_answers)} (section {section_number}): {question}")
        answer, _, _, chunks, original_answer, cited_chunk_ids = retrieve_database_local(question, "en", "labour", chat_model=model_name, hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig, engine=engine, n_results=5, is_remote=False, hardcode_question_section_numbers=[section_number], include_html_in_citations=False)

        # Remove all chunks that are not in the cited_chunk_ids
        cited_chunks = [chunk for chunk in chunks if chunk[2] in cited_chunk_ids]

        # if chunks are empty, keep first one
        if len(cited_chunks) == 0:
            cited_chunks = [chunks[0]]

        if len(cited_chunks) > 1:
            print(f"WARNING: {len(cited_chunks)} chunks found for question {question}")

        document_messages = []

        for document, _, id, _ in cited_chunks:
            document_messages.append({
                'role': 'user',
                'content': f"{id}:\n\n{document}"
            })

        json_results.append({
            "section_number": section_number,
            "document_messages": document_messages,
            "question": question,
            "answer": answer
        })

        # Save to file every question (so we don't lose progress if the script crashes)
        with open(f"experiments/create_qa_datasets/clara_questions_answers_datasets/{filename}", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=4)
            
        print(f"Time taken: {time.time() - start_time} seconds")

    print(f"Saved {len(json_results)} results to {filename}, total time taken: {time.time() - total_start_time} seconds")

if __name__ == "__main__":
    # List of models to use
    models = [
        "gemma3n:latest",
        "gemma3n-finetune", 
        "gemma3n-finetune-2-epochs_questions_answers",
        "gemma3n:e4b-it-q8_0"
    ]
    
    # Loop over each model and create the dataset
    for model in models:
        print(f"Processing with model: {model}")
        create_final_qa_dataset(model) 