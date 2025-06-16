import logging
import re
from typing import List
import streamlit as st
from transformers import PreTrainedTokenizerFast

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
from config import PromptTemplateType
from citation import markdown_post_processing

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

@st.cache_resource(show_spinner=False)
def _retrieve_tokenizer():    
    return PreTrainedTokenizerFast(tokenizer_file=".tokenizers/tokenizer.json")

def process_documents(tokenizer, max_context_length, total_tokens, document_messages):
    included_messages = []

    # Process each document message
    for message in document_messages:
        # Add 24 tokens for message template overhead
        message_tokens = count_tokens(message["content"], tokenizer) + PromptTemplateType.message_template_token_count
        
        # Check if adding this message would exceed the limit
        if total_tokens + message_tokens > max_context_length:
            logging.warning(f"WARNING: Prompt length limit was reached: stopped adding documents at {total_tokens} tokens.")
            break
        
        total_tokens += message_tokens
        included_messages.append(message)

    return total_tokens, included_messages

def manage_max_context_length(num_ctx, 
                              document_messages, 
                              prompt, 
                              prompt_question, 
                              previous_documents, 
                              previous_questions_and_answers, 
                              max_context_length):
    tokenizer = _retrieve_tokenizer()
    included_previous_documents = []
    included_main_documents = []
    total_used_tokens = None
    current_num_ctx = num_ctx
    
    # Truncate the context if it's too long
    if tokenizer is not None and max_context_length is not None:
        # Calculate the tokens used by the prompt + question
        prompt_tokens_count = count_tokens(prompt + prompt_question, tokenizer) + PromptTemplateType.message_template_token_count + 1 # +1 for the start token

        if previous_questions_and_answers is not None:
            for message in previous_questions_and_answers:
                prompt_tokens_count += count_tokens(message["content"], tokenizer) + PromptTemplateType.message_template_token_count
        
        # Keep track of total tokens used
        total_used_tokens = prompt_tokens_count

        # Process each main question document message
        total_used_tokens, included_main_documents = process_documents(
            tokenizer, max_context_length, total_used_tokens, document_messages
        )

        # Process each previous question document message
            # Lower priority than main question documents if the context is too long
        included_previous_documents = []
        if previous_documents is not None:
            total_used_tokens, included_previous_documents = process_documents(
                tokenizer, max_context_length, total_used_tokens, previous_documents
            )

        if current_num_ctx is not None and total_used_tokens > current_num_ctx:
            # Round up to nearest 4096 block, capped at max_context_length
            block_size = 4096
            num_blocks = (total_used_tokens + block_size - 1) // block_size
            new_ctx = min(num_blocks * block_size, max_context_length)
            current_num_ctx = new_ctx

    else:
        included_main_documents = document_messages
        included_previous_documents = previous_documents

    # Reverse the order of the main documents to have the most relevant documents at the end
        # Don't reverse the previous documents because they were already reversed when fetching the chunks
    included_main_documents.reverse()

    all_included_documents = included_previous_documents + included_main_documents

    return all_included_documents, current_num_ctx, total_used_tokens

# Format the metadata + documents to be included as context in the prompt
def format_context_for_prompt(ids, metadata_list, documents_list):
    formatted_metadata = []

    for id, metadata in zip(ids, metadata_list):
        metadata_values = [metadata["hierarchy"], metadata["hyperlink"]]
        if metadata.get("section_number"):
            metadata_values.insert(0, "Section " + metadata["section_number"])

        document_source = id.split('-')[0].lower()
        if document_source == "clc":
            document_source = "Canada Labour Code"
        elif document_source == "clsr":
            document_source = "Canada Labour Standards Regulations"
        elif document_source == "ipg":
            document_source = "Interpretations, Policies and Guidelines"
        else:
            document_source = "canada.ca"

        formatted_metadata_text = "{" + f'"id": "{id}", "type": "{document_source}", "title": "{metadata["title"]}"'
        formatted_metadata_text += f', "section": "{metadata["section_number"] if metadata.get("section_number") else "N/A"}"'
        formatted_metadata_text += "}"

        formatted_metadata.append(formatted_metadata_text)

    formatted_documents = [f'{meta}, "text": "{doc.strip()}"' for meta, doc in zip(formatted_metadata, documents_list)]

    return formatted_documents

# Format the metadata to be displayed in the "metadata" tab of the chatbot
def format_for_metadata_tab_ui(ids_list, metadata_list):
    formatted_metadata = []

    for id, metadata in zip(ids_list, metadata_list):
        metadata_values = [metadata["hierarchy"], metadata["hyperlink"]]
        if metadata.get("section_number"):
            metadata_values.insert(0, "Section " + metadata["section_number"])

        new_entry = {id + " - " + metadata["title"]: str(metadata_values)}
        
        if new_entry not in formatted_metadata:
            formatted_metadata.append(new_entry)

    return formatted_metadata

# Format the documents to be displayed in the "documents" tab of the chatbot
def format_for_documents_tab_ui(ids_list, metadata_list, documents_list):
    formatted_documents = []

    for id, metadata, document in zip(ids_list, metadata_list, documents_list):
        formatted_documents.append(f"{id} - {metadata['title']}:\n\n{document.strip()}")

    final_text = "\n\n\n".join(formatted_documents)
    final_text = markdown_post_processing(final_text)
    
    return final_text

def extract_reference_section_numbers(text: str) -> List[str]:
    # Find all occurrences of "Section(s)" or "Article(s)" followed by numbers, case insensitive
    # Matches patterns like "Section 1.23", "Articles 1, 2 and 3", "Section 1.2, 2.3"
    separators = r'\s*,\s*|\s+and\s+|\s+or\s+|\s+et\s+|\s+ou\s+'
    reference_pattern = f"(?:section|article)s?\\s+((?:\\d+(?:\\.\\d+)?(?:{separators})?)+)"
    matches = re.finditer(reference_pattern, text, re.IGNORECASE)
    
    sections = []
    for match in matches:
        # Split the matched numbers by comma, 'and', 'or', 'et', 'ou'
        number_str = match.group(1)
        numbers = re.split(separators, number_str, flags=re.IGNORECASE)
        sections.extend(num.strip() for num in numbers if num.strip())
    
    # Remove duplicates and sort (using natural string sorting for decimal numbers)
    section_numbers = sorted(list(set(sections)), key=lambda x: float(x))
    return section_numbers

def reprioritize_docs(question_section_numbers, documents, metadata, ids, distances):
    if len(question_section_numbers) == 0:
        return documents, metadata, ids, distances
    
    prioritized = []
    non_prioritized = []
    
    # Go through list in reverse order
    for doc, meta, id, dist in list(zip(documents, metadata, ids, distances)):
        # Split reference numbers into list
        ref_numbers = extract_reference_section_numbers(doc)
        ref_numbers_without_subsections = [number.split('.')[0] for number in ref_numbers]

        is_prioritized = False
        
        # Check if any question numbers match
        for q_num in question_section_numbers:
            if q_num in ref_numbers or q_num in ref_numbers_without_subsections:
                is_prioritized = True # Move this item to front of list
                break
        
        if is_prioritized:
            prioritized.append((doc, meta, id, dist))
        else:
            non_prioritized.append((doc, meta, id, dist))
    
    combined = prioritized + non_prioritized

    if len(combined) == 0:
        print("WARNING: No documents returned, potential error in the prioritization process.")
        return documents, metadata, ids, distances

    # Unzip the reordered list
    docs, metas, ids, dists = zip(*combined)
    return list(docs), list(metas), list(ids), list(dists)