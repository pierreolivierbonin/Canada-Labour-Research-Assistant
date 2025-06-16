from bs4 import BeautifulSoup
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
from config import QuotationsConfig
from citation import verify_and_attribute_quotes, markdown_post_processing

def post_processing(text: str, chunks: list, quotations_mode: bool) -> str:
    # Clean any html from the answer
    formatted_text = BeautifulSoup(text, 'html.parser').get_text()

    if quotations_mode:
        formatted_text = verify_and_attribute_quotes(chunks, formatted_text, QuotationsConfig.threshold_rouge_score, True, False, False)

    return formatted_text

def text_generator(paragraph: str):
    buffer = []
    for letter in paragraph:
        buffer.append(letter)
        if len(buffer) == 3:
            yield ''.join(buffer)
            buffer = []
            time.sleep(0.01)
    if buffer:  # Yield any remaining chunks
        yield ''.join(buffer)
        time.sleep(0.01)

def combine_paragraphs(paragraphs: list):
    return "\n\n".join(paragraphs)

def get_paragraph_generator(stream_generator, chunks: list, quotations_mode: bool):
    paragraphs = []
    unprocessed_paragraphs = []  # New list to store unprocessed paragraphs
    current_chunk = ""

    latest_paragraph_chunks = []
    
    if stream_generator is not None:
        for chunk in stream_generator:
            current_chunk += chunk
            latest_paragraph_chunks += chunk
            
            # Split on double newlines to detect paragraphs
            parts = current_chunk.split("\n\n")
            
            # If we have at least one complete paragraph
            if len(parts) > 1:
                current_paragraph = markdown_post_processing(combine_paragraphs(parts[:-1]))

                # if current paragraph is empty or if it contains only a bullet point, skip it
                if current_paragraph.strip("\n *") == "":
                    current_chunk = ""
                    continue

                previous_paragraphs = paragraphs[:] # clone the list
                previous_unprocessed_paragraphs = unprocessed_paragraphs[:] # clone the unprocessed list

                # Process the complete paragraphs
                formatted_paragraph = post_processing(current_paragraph, chunks, quotations_mode)
                paragraphs.append(formatted_paragraph)
                unprocessed_paragraphs.append(current_paragraph)  # Add unprocessed version
                
                # Keep the last part as the current chunk
                current_chunk = parts[-1]
                
                # Combine processed paragraphs with current chunk
                yield combine_paragraphs(previous_paragraphs), combine_paragraphs(previous_unprocessed_paragraphs), text_generator(current_paragraph)

                latest_paragraph_chunks = []
    
    # Process any remaining text
    if current_chunk:
        remaining_text = markdown_post_processing(current_chunk)
        yield combine_paragraphs(paragraphs), combine_paragraphs(unprocessed_paragraphs), text_generator(remaining_text)

        formatted_paragraph = post_processing(remaining_text, chunks, quotations_mode)
        paragraphs.append(formatted_paragraph)
        unprocessed_paragraphs.append(remaining_text)  # Add unprocessed version
        yield combine_paragraphs(paragraphs), combine_paragraphs(unprocessed_paragraphs), None