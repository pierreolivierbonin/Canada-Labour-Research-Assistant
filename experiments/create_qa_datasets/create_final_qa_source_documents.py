import json
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Add the parent directory to sys.path

from config import OllamaRAGConfig, ChatbotInterfaceConfig
from tools import retrieve_database_local
import time

def main():
    engine = "ollama"
    
    # Read questions_answers.json
    with open("experiments/create_qa_datasets/questions_answers.json", "r", encoding="utf-8") as f:
        questions_answers = json.load(f)

    json_results = []
    total_start_time = time.time()
    filename = "questions_source_documents.json"
    
    for x, question_obj in enumerate(questions_answers):
        question = question_obj["question"]
        section_number = question_obj["section_number"]
        start_time = time.time()

        print(f"Question {x+1}/{len(questions_answers)} (section {section_number}): {question}")
        answer, _, _, chunks, original_answer, cited_chunk_ids = retrieve_database_local(question, "en", "labour", chat_model=ChatbotInterfaceConfig.default_model_local, hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig, engine=engine, n_results=5, is_remote=False, hardcode_question_section_numbers=[section_number], include_html_in_citations=False)

        # Can comment out the local llm call inside retrieve_database_local to get the chunks only
        # _, _, chunks = retrieve_database_local(question, "en", "labour", chat_model=ChatbotInterfaceConfig.default_model_local, hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig, engine=engine, n_results=5, is_remote=False, hardcode_question_section_numbers=[section_number], include_html_in_citations=False)

        original_documents = [chunk[0] for chunk in chunks]

        json_results.append({
            "section_number": section_number,
            "question": question,
            "original_documents": original_documents
        })

    # Save to file every question (so we don't lose progress if the script crashes)
    with open(f"experiments/create_qa_datasets/questions_source_documents_2.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)
        
    print(f"Time taken: {time.time() - start_time} seconds")

    print(f"Saved {len(json_results)} results to {filename}, total time taken: {time.time() - total_start_time} seconds")

if __name__ == "__main__":
    main() 