import json
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import vLLMRAGConfig, vLLMChatbotInterfaceConfig
from tools import retrieve_database_local

def main():
    # Read questions_answers.json
    with open("questions_answers.json", "r", encoding="utf-8") as f:
        questions_answers = json.load(f)

    json_results = []
    
    for x, question_obj in enumerate(questions_answers):
        question = question_obj["question"]
        section_number = question_obj["section_number"]

        #question = f"The following question is related to section {section_number}\n{question}"

        print(f"Question {x+1}/{len(questions_answers)} (section {section_number}): {question}")
        answer, _, _, chunks, original_answer, cited_chunk_ids = retrieve_database_local(question, "en", "labour", chat_model=vLLMChatbotInterfaceConfig.default_model_local, hyperparams=vLLMRAGConfig.HyperparametersAccuracyConfig, engine="vllm", n_results=5, is_remote=False, hardcode_question_section_numbers=[section_number], include_html_in_citations=False)

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

    with open("final_questions_answers_fine_tuned_model.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)

    print(f"Saved {len(json_results)} results to final_questions_answers.json")

if __name__ == "__main__":
    main() 