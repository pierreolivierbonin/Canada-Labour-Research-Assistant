import json
import pandas as pd
import torch
from unsloth import FastLanguageModel
from transformers.generation.streamers import TextStreamer
import re
from tqdm import tqdm
from datetime import datetime

model_name = "lora_model_mcqa_e_32_a_32_e_5" #"unsloth/Llama-3.2-3B-Instruct"
model_name_split = model_name.split("/")
model_name_no_path = model_name_split[1] if len(model_name_split) > 1 else model_name

dataset_path = "experiments/eval/clc_bonito_dataset_mcqa/data.jsonl"
dataset_name = dataset_path.split("/")[0]
clc_data_path = "experiments/create_qa_datasets/clc_dataset/clc_data.csv"

def load_model():
    """Load the LoRA model for inference"""
    max_seq_length = None
    dtype = None
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= model_name,  # Using the MCQA model
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def load_jsonl_data(file_path):
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def load_csv_data(file_path):
    """Load CSV data"""
    return pd.read_csv(file_path)

def find_matching_section(context, csv_data):
    """Find the section_number for the given context in the CSV data"""
    # Look for exact match first
    exact_match = csv_data[csv_data['text'] == context]
    if not exact_match.empty:
        return exact_match.iloc[0]['section_number']
    
    # If no exact match, look for partial match (context might be a substring)
    for idx, row in csv_data.iterrows():
        context_no_spaces = context.replace(" ", "")
        text_no_spaces = row['text'].replace(" ", "")
        if context_no_spaces in text_no_spaces or text_no_spaces in context_no_spaces:
            return row['section_number']
    
    # If still no match, try to extract section number from context directly
    section_match = re.search(r'Section (\d+)', context)
    if section_match:
        return int(section_match.group(1))
    
    return None

def replace_context(instruction, context):
    """Replace {{context}} with the correct context"""
    return instruction.replace("{{context}}", context)
    
def replace_context_placeholder(instruction, section_number):
    """Replace {{context}} placeholder with section number reference"""
    if section_number is not None:
        section_ref = f"Related to section {section_number} of the Canada Labour Code (CLC)"
        return instruction.replace("{{context}}", section_ref)
    else:
        # If no section found, remove the placeholder
        return instruction.replace("{{context}}", "")

def query_model(model, tokenizer, instruction):
    """Query the model with the given instruction"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about the Canada Labour Code (CLC). You should directly answer the question based on the context provided, without any additional explanation or commentary. Ex: User: Having read the above passage, choose the right answer to the following question (choices are 2012 or 2014). Answer: 2014"},
        {"role": "user", "content": instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=256,
            do_sample=False,  # Use greedy decoding for consistency
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def clean_predicted_expected(predicted, expected):
    """Clean predicted and expected strings"""
    clean_predicted = predicted.strip("- ").lower()
    clean_expected = expected.strip("- ").lower()

    # if expecteed = 1 char (multiple choice), only compare with the first char of predicted (in case first char was the letter of the choice, followed by the actual text of the choice)
    is_1_char_expected = False
    if len(clean_expected) == 1:
        clean_predicted = clean_predicted[0]
        is_1_char_expected = True

    return clean_predicted, clean_expected, is_1_char_expected

def calculate_exact_match_score(predicted, expected):
    cleaned_predicted, cleaned_expected, is_1_char_expected = clean_predicted_expected(predicted, expected)
    """Calculate exact match score"""
    return 1 if cleaned_predicted == cleaned_expected else 0, is_1_char_expected

def calculate_partial_match_score(predicted, expected):
    """Calculate partial match score (if predicted answer is contained in expected or vice versa)"""
    cleaned_predicted, cleaned_expected, _ = clean_predicted_expected(predicted, expected)
    
    # if pred_lower == exp_lower:
    #     return 1.0
    if cleaned_predicted in cleaned_expected or cleaned_expected in cleaned_predicted:
        return 1.0
    else:
        return 0.0

def save_results_to_file(results_dict, include_context):
    filepath = "experiments/eval/results/" + dataset_name + "_" + model_name_no_path + ("_with_context" if include_context else "") + ".txt"

    """Save evaluation results to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CLC BONITO DATASET MCQA EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Reference: {clc_data_path}\n")
        f.write("-"*60 + "\n\n")
        
        f.write("SUMMARY METRICS:\n")
        f.write(f"Total items processed: {results_dict['total_items']}\n")
        f.write(f"Items without section matches: {results_dict['no_section_matches']}\n")
        f.write(f"Exact matches: {results_dict['exact_matches']}\n")
        f.write(f"Partial matches: {results_dict['partial_matches']}\n\n")
        
        f.write("ACCURACY SCORES:\n")
        f.write(f"Exact match accuracy: {results_dict['exact_accuracy']:.2f}%\n")
        f.write(f"Partial match accuracy: {results_dict['partial_accuracy']:.2f}%\n")
        f.write(f"Average exact score: {results_dict['avg_exact_score']:.4f}\n")
        f.write(f"Average partial score: {results_dict['avg_partial_score']:.4f}\n\n")
        
        f.write("DETAILED BREAKDOWN:\n")
        f.write(f"Success rate (items with section matches): {results_dict['success_rate']:.2f}%\n")
        f.write(f"Coverage (items successfully processed): {results_dict['coverage']:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FINAL SCORE: {:.2f}% (Exact Match Accuracy)\n".format(results_dict['exact_accuracy']))
        f.write("="*60 + "\n")

    return filepath

def main():
    include_context = True

    print("Loading model...")
    model, tokenizer = load_model()
    
    print("Loading datasets...")
    # Load JSONL data
    jsonl_data = load_jsonl_data(dataset_path)
    print(f"Loaded {len(jsonl_data)} items from JSONL")
    
    # Load CSV data
    csv_data = load_csv_data(clc_data_path)
    print(f"Loaded {len(csv_data)} items from CSV")
    
    # Initialize scoring
    total_items = 0
    exact_matches = 0
    partial_matches = 0
    total_exact_score = 0
    total_partial_score = 0
    
    # Track items without section matches
    no_section_matches = 0
    
    print("Starting evaluation...")
    
    # Process each item in the JSONL data
    for i, item in enumerate(tqdm(jsonl_data, desc="Evaluating")):
        try:
            # Extract required fields
            context = item.get('context', '')
            instruction = item.get('instruction', '')
            expected_output = item.get('output', '')
            
            if not instruction or not expected_output:
                print(f"Skipping item {i}: missing instruction or output")
                continue
            
            # Find matching section number
            section_number = find_matching_section(context, csv_data)
            if section_number is None:
                no_section_matches += 1
                print(f"Warning: No section match found for item {i}")
            
            # Replace context placeholder with section reference
            if include_context:
                processed_instruction = replace_context(instruction, context)
            else:
                processed_instruction = replace_context_placeholder(instruction, section_number)
            
            # Query the model
            predicted_output = query_model(model, tokenizer, processed_instruction)
            
            # Calculate scores
            exact_score, is_1_char_expected = calculate_exact_match_score(predicted_output, expected_output)

            # If only 1 single char is expected, don't calculate partial score
            if not is_1_char_expected:
                partial_score = calculate_partial_match_score(predicted_output, expected_output)
            else:
                partial_score = exact_score
            
            total_exact_score += exact_score
            total_partial_score += partial_score
            
            if exact_score == 1:
                exact_matches += 1
            if partial_score > 0:
                partial_matches += 1
            
            total_items += 1
            
            # Print progress every 50 items
            if (i + 1) % 50 == 0:
                current_exact_accuracy = (total_exact_score / total_items) * 100
                current_partial_accuracy = (total_partial_score / total_items) * 100
                print(f"Progress: {i+1}/{len(jsonl_data)} - "
                      f"Exact: {current_exact_accuracy:.2f}% - "
                      f"Partial: {current_partial_accuracy:.2f}%")
            
            # Print first few examples for debugging
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Section: {section_number}")
                print(f"Instruction: {processed_instruction[:200]}...")
                print(f"Expected: {expected_output}")
                print(f"Predicted: {predicted_output}")
                print(f"Exact Match: {exact_score}")
                print(f"Partial Score: {partial_score}")
                print("-" * 50)
        
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            continue
    
    # Calculate final scores
    if total_items > 0:
        exact_accuracy = (total_exact_score / total_items) * 100
        partial_accuracy = (total_partial_score / total_items) * 100
        success_rate = ((total_items - no_section_matches) / total_items) * 100
        coverage = (total_items / len(jsonl_data)) * 100
        
        # Prepare results dictionary
        results = {
            'total_items': total_items,
            'no_section_matches': no_section_matches,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'exact_accuracy': exact_accuracy,
            'partial_accuracy': partial_accuracy,
            'avg_exact_score': total_exact_score / total_items,
            'avg_partial_score': total_partial_score / total_items,
            'success_rate': success_rate,
            'coverage': coverage
        }
        
        # Save results to file
        print(f"\nSaving results to eval...")
        eval_filepath = save_results_to_file(results, include_context)
        print("Results saved successfully!")

        # PRint the results from the file
        with open(eval_filepath, "r", encoding="utf-8") as f:
            print(f.read())
        
    else:
        print("No items were successfully processed!")
        # Save error state to file
        error_results = {
            'total_items': 0,
            'no_section_matches': 0,
            'exact_matches': 0,
            'partial_matches': 0,
            'exact_accuracy': 0.0,
            'partial_accuracy': 0.0,
            'avg_exact_score': 0.0,
            'avg_partial_score': 0.0,
            'success_rate': 0.0,
            'coverage': 0.0
        }
        save_results_to_file(error_results, include_context)

if __name__ == "__main__":
    main() 