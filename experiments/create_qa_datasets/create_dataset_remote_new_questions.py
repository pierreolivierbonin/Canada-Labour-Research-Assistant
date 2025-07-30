from datasets import load_dataset, Dataset
import time
import random

import requests
import toml
import json
import re

from create_dataset_remote import read_secrets, get_llm_answer_remote, extract_json_from_response

def load_existing_questions(file_path):
    """Load existing questions from questions_answers.json and create a mapping by section_number"""
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
        
        # Create a dictionary mapping section_number to existing questions
        section_to_question = {}
        for item in existing_data:
            section_number = item["section_number"]
            question = item["question"]
            section_to_question[section_number] = question
        
        return section_to_question
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Will generate questions without avoiding duplicates.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse {file_path}. {str(e)}")
        return {}

def main():
    start_time = time.time()

    # Load existing questions to avoid duplicates
    existing_questions = load_existing_questions("experiments/create_qa_datasets/questions_answers.json")

    # Create a dataset based on clc_dataset/data.csv, where the text column is the unannotated text
    clc_dataset = load_dataset("csv", data_files="experiments/create_qa_datasets/clc_dataset/data.csv", split="train", streaming=False) #.select(range(10))

    # Select 100 random sections
    clc_dataset = clc_dataset.select(random.sample(range(len(clc_dataset)), 100))

    questions_answers = []

    # Loop over all the sections of the CLC (except the first one)
    for i in range(1, len(clc_dataset)):
        section_number = clc_dataset[i]['section_number']
        text = clc_dataset[i]['text']

        # Check if we have an existing question for this section
        existing_question = existing_questions.get(section_number)
        
        if existing_question:
            # If there's an existing question, ask for a different one
            user_prompt = f"Generate a NEW and DIFFERENT question based on the given section of the Canada Labour Code (CLC). It should be possible to quote passages from the section given to answer the question. The question should not refer to specific parts of the section, but rather to the general idea of the section. Then, give a short answer to the question.\n\nIMPORTANT: The question must be DIFFERENT from this existing question: \"{existing_question}\"\n\nYour response should be fully in json, in the following format: {{\"explanation\": \"[short explanation for why the question was chosen and how it differs from the existing question]\", \"question\": \"[NEW question that is different from the existing one]\", \"answer\": \"[short answer]\"}}"
        else:
            print(f"WARNING: No existing question for section {section_number}, skipping...")
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates questions based on sections of the Canada Labour Code (CLC)."},
            {"role": "user", "content": text},
            {"role": "user", "content": user_prompt}
        ]
        
        # Ask the LLM to generate a question and answer based on the text
        json_response = get_llm_answer_remote("meta-llama/Llama-4-Scout-17B-16E-Instruct", messages)
        json_response = extract_json_from_response(json_response)
        
        if json_response and 'question' in json_response and 'answer' in json_response:
            question = json_response['question']
            answer = json_response['answer']

            questions_answers.append({
                "section_number": section_number,
                "question": question,
                "answer": answer
            })

            print(f"Section {section_number}: Generated new question")
            if existing_question:
                print(f"  Existing: {existing_question[:100]}...")
                print(f"  New: {question[:100]}...")
            else:
                print(f"  Question: {question[:100]}...")
        else:
            print(f"Error: Failed to generate valid question for section {section_number}")

        print(f"Total questions generated so far: {len(questions_answers)}")

    # Save the questions and answers to a json file
    with open("experiments/create_qa_datasets/new_questions_answers.json", "w") as f:
        json.dump(questions_answers, f)

    print(f"Time taken: {time.time() - start_time} seconds")

# main 
if __name__ == "__main__":
    main()