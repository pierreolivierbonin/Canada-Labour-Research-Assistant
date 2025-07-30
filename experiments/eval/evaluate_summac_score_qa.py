import json
import sys
import os
from dataclasses import dataclass
import statistics
import time

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.evaluation import summac_consistency_detection
from rouge import Rouge

@dataclass
class EvaluationConfig:
    summac = {
        "models": ["vitc"],
        "bins": 'percentile',
        'granularity': 'sentence',
        'nli_labels': 'e',
        'device': 'cuda',
        'start_file': '.models/summac_conv_vitc_sent_perc_e.bin',
        'agg': 'mean'
    }

def combine_documents(document_messages):
    """Combine all document messages into a single text string."""
    combined_text = ""
    for content in document_messages:
        # Remove the document ID prefix (e.g., "CLC-2:\n\n")
        if ":\n\n" in content:
            content = content.split(":\n\n", 1)[1]
        combined_text += content + "\n\n"
    return combined_text.strip()

def calculate_rouge_scores(rouge_evaluator, reference_text, candidate_text):
    """ Calculate ROUGE scores between reference and candidate text. """
    
    try:
        # Clean texts to avoid empty strings
        if not reference_text.strip() or not candidate_text.strip():
            return None
        
        scores = rouge_evaluator.get_scores(candidate_text, reference_text)
        
        # Extract the scores (get_scores returns a list with one dict)
        rouge_scores = scores[0]
        
        return {
            'rouge-1': {
                'precision': rouge_scores['rouge-1']['p'],
                'recall': rouge_scores['rouge-1']['r'],
                'f1': rouge_scores['rouge-1']['f']
            },
            'rouge-2': {
                'precision': rouge_scores['rouge-2']['p'],
                'recall': rouge_scores['rouge-2']['r'],
                'f1': rouge_scores['rouge-2']['f']
            },
            'rouge-l': {
                'precision': rouge_scores['rouge-l']['p'],
                'recall': rouge_scores['rouge-l']['r'],
                'f1': rouge_scores['rouge-l']['f']
            }
        }
    except Exception as e:
        print(f"Error calculating ROUGE scores: {str(e)}")
        return None

def evaluate_qa_dataset(questions_source_file, answers_file, output_prefix):
    """ Evaluate QA pairs using summac consistency detection and ROUGE scores. """
    
    print(f"Loading source documents from {questions_source_file}")
    with open(questions_source_file, 'r', encoding='utf-8') as f:
        questions_source = json.load(f)
    
    print(f"Loading answers from {answers_file}")
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    
    # Initialize ROUGE evaluator
    rouge = Rouge()
    
    # Create lookup dictionary for answers by section_number
    answers_lookup = {item['section_number']: item for item in answers_data}
    
    # Results storage
    individual_results = []
    summac_scores = []
    rouge_1_f1_scores = []
    rouge_2_f1_scores = []
    rouge_l_f1_scores = []
    failed_evaluations = []
    
    total_questions = len(questions_source)
    print(f"Starting evaluation of {total_questions} questions...")
    
    start_time = time.time()
    
    for idx, source_item in enumerate(questions_source):
        section_number = source_item['section_number']
        question = source_item['question']
        
        print(f"Processing question {idx + 1}/{total_questions} (section {section_number})")
        
        # Find matching answer
        if section_number not in answers_lookup:
            print(f"Warning: No answer found for section {section_number}")
            failed_evaluations.append({
                'section_number': section_number,
                'question': question,
                'error': 'No matching answer found'
            })
            continue
        
        answer_item = answers_lookup[section_number]
        answer = answer_item['answer']
        
        # Combine source documents
        documents = combine_documents(source_item['original_documents'])
        
        if not documents.strip():
            print(f"Warning: No documents found for section {section_number}")
            failed_evaluations.append({
                'section_number': section_number,
                'question': question,
                'error': 'No source documents found'
            })
            continue
        
        try:
            # Evaluate using summac
            summac_score = summac_consistency_detection(
                EvaluationConfig.summac,
                documents,
                answer
            )
            
            # Extract the summac score value
            summac_score_value = summac_score[0] if isinstance(summac_score, list) and len(summac_score) > 0 else summac_score

            if "scores" in summac_score_value and len(summac_score_value["scores"]) > 0:
                if len(summac_score_value["scores"]) > 1:
                    print(f"Warning: Multiple summac scores found for section {section_number}: {summac_score_value['scores']}")
                summac_score_value = summac_score_value["scores"][0]
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(rouge, documents, answer)
            
            result_item = {
                'section_number': section_number,
                'question': question,
                'answer': answer,
                'documents_preview': documents[:200] + "..." if len(documents) > 200 else documents,
                'summac_score': summac_score_value,
                'raw_summac_score': summac_score,
                'rouge_scores': rouge_scores
            }
            
            individual_results.append(result_item)
            summac_scores.append(summac_score_value)
            
            # Store ROUGE F1 scores for summary statistics
            if rouge_scores:
                rouge_1_f1_scores.append(rouge_scores['rouge-1']['f1'])
                rouge_2_f1_scores.append(rouge_scores['rouge-2']['f1'])
                rouge_l_f1_scores.append(rouge_scores['rouge-l']['f1'])
                
                print(f"Section {section_number} - Summac: {summac_score_value:.4f}, "
                      f"ROUGE-L F1: {rouge_scores['rouge-l']['f1']:.4f}")
            else:
                print(f"Section {section_number} - Summac: {summac_score_value:.4f}, ROUGE: Failed")
            
        except Exception as e:
            print(f"Error evaluating section {section_number}: {str(e)}")
            failed_evaluations.append({
                'section_number': section_number,
                'question': question,
                'error': str(e)
            })
    
    # Calculate summary statistics
    summary = {
        'total_questions': total_questions,
        'successful_evaluations': len(summac_scores),
        'failed_evaluations': len(failed_evaluations),
        'summac_metrics': {
            'average_score': statistics.mean(summac_scores),
            'median_score': statistics.median(summac_scores),
            'min_score': min(summac_scores),
            'max_score': max(summac_scores),
            'standard_deviation': statistics.stdev(summac_scores) if len(summac_scores) > 1 else 0
        },
        'evaluation_time_seconds': time.time() - start_time,
        'dataset_evaluated': answers_file
    }
    
    # Add ROUGE statistics if available
    if rouge_1_f1_scores:
        summary['rouge_metrics'] = {
            'rouge-1_f1': {
                'average': statistics.mean(rouge_1_f1_scores),
                'median': statistics.median(rouge_1_f1_scores),
                'min': min(rouge_1_f1_scores),
                'max': max(rouge_1_f1_scores),
                'std': statistics.stdev(rouge_1_f1_scores) if len(rouge_1_f1_scores) > 1 else 0
            },
            'rouge-2_f1': {
                'average': statistics.mean(rouge_2_f1_scores),
                'median': statistics.median(rouge_2_f1_scores),
                'min': min(rouge_2_f1_scores),
                'max': max(rouge_2_f1_scores),
                'std': statistics.stdev(rouge_2_f1_scores) if len(rouge_2_f1_scores) > 1 else 0
            },
            'rouge-l_f1': {
                'average': statistics.mean(rouge_l_f1_scores),
                'median': statistics.median(rouge_l_f1_scores),
                'min': min(rouge_l_f1_scores),
                'max': max(rouge_l_f1_scores),
                'std': statistics.stdev(rouge_l_f1_scores) if len(rouge_l_f1_scores) > 1 else 0
            }
        }

    # Check if the directory exists, if not create it
    os.makedirs("experiments/eval/summac_rouge_results", exist_ok=True)
    
    # Save individual results
    individual_results_file = f"experiments/eval/summac_rouge_results/{output_prefix}_summac_rouge_results.json"
    with open(individual_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'individual_results': individual_results,
            'failed_evaluations': failed_evaluations
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Successful evaluations: {summary['successful_evaluations']}")
    print(f"Failed evaluations: {summary['failed_evaluations']}")
    
    print(f"Results saved to: {individual_results_file}")
    
    return summary

def main():
    """Main function to run the evaluation."""
    
    # File paths
    questions_source_file = "experiments/create_qa_datasets/questions_source_documents.json"
    output_prefix = "gemma3n-finetune-2-epochs_questions_answers_q8"
    answers_file = f"experiments/create_qa_datasets/{output_prefix}.json"
    
    # Check if files exist
    if not os.path.exists(questions_source_file):
        print(f"Error: Source file not found: {questions_source_file}")
        return
    
    if not os.path.exists(answers_file):
        print(f"Error: Answers file not found: {answers_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("experiments/eval", exist_ok=True)
    
    # Run evaluation
    try:
        summary = evaluate_qa_dataset(questions_source_file, answers_file, output_prefix)
        print("\nEvaluation completed successfully!")
        return summary
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
