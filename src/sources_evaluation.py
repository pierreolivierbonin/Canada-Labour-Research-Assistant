import pandas as pd
from typing import Set

def format_url(urls_text: str) -> list[str]:
    urls_text = urls_text.replace("\r", "").replace("\n", "")
    urls = urls_text.split(';')
    return [url.split('#')[0].strip() for url in urls] # Remove the # and the text after from the url

# Could be float or str
def format_toc_sections(section) -> list[str]:
    formatted_sections = str(section).split(';')
    return [section.split('.')[0] for section in formatted_sections] # If section = number + ".0" remove the ".0"

# Extract all sources from a row, including URLs and section numbers.
def get_all_sources(row: pd.Series) -> Set[str]:
    sources = set()
    
    if pd.notna(row.get('sources')):
        sources.update(format_url(row['sources']))
    
    if pd.notna(row.get('sections_clc')):
        sources.update(format_toc_sections(row['sections_clc']))
    
    if pd.notna(row.get('sections_clsr')):
        sources.update(format_toc_sections(row['sections_clsr']))
        
    return sources

# "Calculate precision, recall, F1 score, and correct sources ratio.
def calculate_metrics(golden_sources: Set[str], retrieved_sources: Set[str]) -> tuple[float, float, float, float, float, float, str]:
    if not retrieved_sources:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, fr"0 \ {len(golden_sources)}"
    
    true_positives = len(golden_sources.intersection(retrieved_sources))
    
    precision = true_positives / len(retrieved_sources) if retrieved_sources else 0
    recall = true_positives / len(golden_sources) if golden_sources else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    total_sources_retrieved = len(retrieved_sources)
    total_sources_golden = len(golden_sources)
    
    ratio = fr"{true_positives} \ {total_sources_golden}"
    
    return total_sources_retrieved, true_positives, total_sources_golden, precision, recall, f1, ratio

def evaluate_dataset_sources(golden_qa: pd.DataFrame, retrieved_qa: pd.DataFrame, qa_citations: pd.DataFrame, output_path: str):
    nb_retrieved_sources = []
    nb_correct_sources = []
    nb_golden_sources = []
    precisions = []
    recalls = []
    f1_scores = []
    ratios = []
    
    # Process each evaluation question
    for _, retrieved_row in retrieved_qa.iterrows():
        # Get corresponding gold standard row
        gold_row = golden_qa[golden_qa['golden_question_id'] == retrieved_row['golden_question_id']].iloc[0]
        
        # Get gold sources
        golden_sources = get_all_sources(gold_row)
        
        # Get evaluation sources from qa_citations
        qa_citations_subset = qa_citations[qa_citations['question_id'] == retrieved_row['question_id']]
        retrieved_sources = set()
        
        # Collect all sources from qa_citations
        for _, citation in qa_citations_subset.iterrows():
            if pd.notna(citation.get('url')):
                retrieved_sources.update(format_url(citation['url']))
            if pd.notna(citation.get('sections_clc')):
                retrieved_sources.update(format_toc_sections(citation['sections_clc']))
            if pd.notna(citation.get('sections_clsr')):
                retrieved_sources.update(format_toc_sections(citation['sections_clsr']))
        
        # Calculate metrics
        retrieved_sources_nb, correct_sources_nb, golden_sources_nb, precision, recall, f1, ratio = calculate_metrics(golden_sources, retrieved_sources)
        
        nb_retrieved_sources.append(retrieved_sources_nb)
        nb_correct_sources.append(correct_sources_nb)
        nb_golden_sources.append(golden_sources_nb)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ratios.append(ratio)

    # Remove answer column
    retrieved_qa = retrieved_qa.drop(columns=['answer'])
    
    # Add metrics to retrieved_qa DataFrame
    retrieved_qa["retrieved_sources"] = nb_retrieved_sources
    retrieved_qa["correct_sources"] = nb_correct_sources
    retrieved_qa["golden_sources"] = nb_golden_sources
    retrieved_qa['correct_ratio'] = ratios
    retrieved_qa['precision'] = [round(p, 2) for p in precisions]
    retrieved_qa['recall'] = [round(r, 2) for r in recalls]
    retrieved_qa['f1_score'] = [round(f, 2) for f in f1_scores]
    
    # Add totals row with averages
    totals_row = pd.DataFrame({
        'question_id': [''],
        'golden_question_id': [''],
        'question': [''],
        'retrieved_sources': '',
        'correct_sources': '',
        'golden_sources': '',
        'correct_ratio': ['MEAN'],
        'precision': [round(retrieved_qa['precision'].mean(), 2)],
        'recall': [round(retrieved_qa['recall'].mean(), 2)],
        'f1_score': [round(retrieved_qa['f1_score'].mean(), 2)]
    })
    
    # Concatenate the totals row to retrieved_qa
    retrieved_qa = pd.concat([retrieved_qa, totals_row], ignore_index=True)
    
    # Save updated retrieved_qa to CSV
    retrieved_qa.to_csv(output_path, index=False)