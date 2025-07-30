import os
import json
import sys
import statistics
import time

# Add the src directory to the path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.citation import verify_and_attribute_quotes

def prepare_chunks_from_documents(documents):
    """
    Convert source documents to the format expected by verify_and_attribute_quotes.
    Expected format: [(chunk_text, chunk_url, chunk_id, chunk_title), ...]
    """
    chunks = []
    for idx, doc_content in enumerate(documents):
        # Extract document ID if present (format: "DOC-ID:\n\nContent")
        if ":\n\n" in doc_content:
            doc_id, content = doc_content.split(":\n\n", 1)
            chunk_id = doc_id.strip()
            chunk_title = f"Document {chunk_id}"
        else:
            chunk_id = f"DOC-{idx}"
            chunk_title = f"Document {idx}"
            content = doc_content
        
        # Create a dummy URL since we don't have real URLs
        chunk_url = f"#doc-{chunk_id.lower()}"
        
        chunks.append((content, chunk_url, chunk_id, chunk_title))
    
    return chunks

def evaluate_hallucination_rate(questions_source_file, answers_file, output_prefix):
    """
    Evaluate hallucination rate using citation matching.
    """
    print(f"Loading source documents from {questions_source_file}")
    with open(questions_source_file, 'r', encoding='utf-8') as f:
        questions_source = json.load(f)
    
    print(f"Loading answers from {answers_file}")
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    
    # Create lookup dictionary for answers by section_number
    answers_lookup = {item['section_number']: item for item in answers_data}
    
    # Results storage
    individual_results = []
    all_citation_scores = []
    questions_with_no_citations = 0
    citations_below_threshold = 0
    citations_not_perfect = 0  # New metric for citations with score != 1
    total_citations = 0
    
    total_questions = len(questions_source)
    print(f"Starting hallucination evaluation of {total_questions} questions...")
    
    start_time = time.time()
    
    for idx, source_item in enumerate(questions_source):
        section_number = source_item['section_number']
        question = source_item['question']
        
        print(f"Processing question {idx + 1}/{total_questions} (section {section_number})")
        
        # Find matching answer
        if section_number not in answers_lookup:
            print(f"Warning: No answer found for section {section_number}")
            continue
        
        answer_item = answers_lookup[section_number]
        answer = answer_item['answer']
        
        # Prepare source documents as chunks
        documents = source_item['original_documents']
        if not documents:
            print(f"Warning: No documents found for section {section_number}")
            continue
        
        chunks = prepare_chunks_from_documents(documents)
        
        # Use citation system with threshold=0 (always try to find closest match)
        modified_answer, cited_chunk_ids, citation_scores = verify_and_attribute_quotes(
            chunks, answer, threshold=0.0, include_html=False, include_attribution=False, include_complete_sentence=False
        )
        
        # Calculate metrics for this question
        question_has_citations = len(citation_scores) > 0
        if not question_has_citations:
            questions_with_no_citations += 1
        
        question_citations_below_threshold = 0
        for citation in citation_scores:
            total_citations += 1
            all_citation_scores.append(citation['best_score'])
            if citation['best_score'] < 0.5:
                question_citations_below_threshold += 1
                citations_below_threshold += 1
            if citation['best_score'] != 1.0: # Check for non-perfect matches
                citations_not_perfect += 1
        
        result_item = {
            'section_number': section_number,
            'question': question,
            'answer': answer,
            'modified_answer': modified_answer,
            'cited_chunk_ids': cited_chunk_ids,
            'citation_scores': citation_scores,
            'num_citations': len(citation_scores),
            'citations_below_threshold': question_citations_below_threshold,
            'has_citations': question_has_citations,
            'avg_citation_score': statistics.mean([c['best_score'] for c in citation_scores]) if citation_scores else 0.0,
            'documents_count': len(documents)
        }
        
        individual_results.append(result_item)
        
        print(f"Section {section_number} - Citations: {len(citation_scores)}, "
                f"Avg Score: {result_item['avg_citation_score']:.4f}, "
                f"Below 0.5: {question_citations_below_threshold}")
        
    # Calculate summary statistics
    successful_evaluations = len(individual_results)
    
    summary = {
        'total_questions': total_questions,
        'successful_evaluations': successful_evaluations,
        'citation_metrics': {
            'total_citations': total_citations,
            'questions_with_no_citations': questions_with_no_citations,
            'citations_below_threshold_0_5': citations_below_threshold,
            'citations_not_perfect': citations_not_perfect,
            'percentage_questions_no_citations': (questions_with_no_citations / total_questions) * 100,
            'percentage_citations_below_0_5': (citations_below_threshold / total_citations) * 100 if total_citations > 0 else 0,
            'percentage_non_perfect_citations': (citations_not_perfect / total_citations) * 100 if total_citations > 0 else 0,
            'average_citation_score': statistics.mean(all_citation_scores),
            'median_citation_score': statistics.median(all_citation_scores),
            'min_citation_score': min(all_citation_scores),
            'max_citation_score': max(all_citation_scores),
            'std_citation_score': statistics.stdev(all_citation_scores) if len(all_citation_scores) > 1 else 0
        },
        'evaluation_time_seconds': time.time() - start_time,
        'dataset_evaluated': answers_file,
        'citation_threshold_used': 0.0
    }
    
    # Check if the directory exists, if not create it
    os.makedirs("experiments/eval/hallucination_rate_results", exist_ok=True)

    # Save results
    results_file = f"experiments/eval/hallucination_rate_results/{output_prefix}_hallucination_results.json" 
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'individual_results': individual_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Hallucination Evaluation Complete ===")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Successful evaluations: {summary['successful_evaluations']}")
    
    print(f"Results saved to: {results_file}")
    
    return summary

def main():
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
        summary = evaluate_hallucination_rate(questions_source_file, answers_file, output_prefix)
        print("\nHallucination evaluation completed successfully!")
        return summary
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
