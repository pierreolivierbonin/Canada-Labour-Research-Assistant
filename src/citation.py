import re
import string
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
from config import QuotationsConfig

# Function to clean text by removing punctuation and converting to lowercase
def clean_text(text):
    translator = str.maketrans('', '', string.punctuation)
    cleaned = text.translate(translator).lower()
    return [word for word in cleaned.split() if word]

# Calculate ROUGE-L score between reference and candidate texts
def calculate_rouge_L_score(reference_words, candidate_words, lcs_length):
    """
    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
    is a metric that measures the similarity between two texts based on their longest common
    subsequence (LCS). The score is calculated as follows:
    
    1. Precision = LCS length / length of candidate text
    2. Recall = LCS length / length of reference text
    3. F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    
    The LCS is the longest sequence of words that appears in both texts in the same order,
    but not necessarily consecutively. A higher ROUGE-L score indicates better similarity
    between the texts.
    
    Args:
        reference_words: List of words in the reference text
        candidate_words: List of words in the candidate text
        lcs_length: Length of the longest common subsequence between the texts
        
    Returns:
        float: ROUGE-L F1 score between 0 and 1, where 1 indicates perfect match
    
    Source: https://aclanthology.org/P04-1077.pdf
    """
    # Calculate precision and recall
    precision = lcs_length / len(candidate_words) if candidate_words else 0
    recall = lcs_length / len(reference_words) if reference_words else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        rougeL_score = 2 * (precision * recall) / (precision + recall)
    else:
        rougeL_score = 0
        
    return rougeL_score

# Calculate ROUGE-L score between reference and candidate texts
def calculate_LCS_and_rouge_L_score(reference, candidate):
    # Clean and tokenize the texts
    reference_words = clean_text(reference)
    candidate_words = clean_text(candidate)
    
    if not reference_words or not candidate_words:
        return 0.0
    
    # Find length of LCS
    lcs_length, _, _ = find_lcs_length(reference_words, candidate_words)
    rougeL_score = calculate_rouge_L_score(reference_words, candidate_words, lcs_length)
        
    return rougeL_score

# Find the length and positions of Longest Common Subsequence between two word lists
def find_lcs_length(reference_words, candidate_words, is_find_optimal_lcs = False) -> tuple[int, int, int]:
    m, n = len(reference_words), len(candidate_words)
    
    # Create Dynamic Programming table for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table containing the length of the LCS for each possible position
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If the words matches, increase the length of the LCS by 1
            if reference_words[i - 1] == candidate_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # If the words don't match, the LCS length is the max of the length of the LCS in the up or left cell
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0, 0, 0

    # Prioritize sequences that occurs earlier in the text
    if is_find_optimal_lcs:
        # Find the earliest ending position in str1 that gives the maximum LCS
        earliest_end = None
        
        # Try each possible ending position
        for end_i in range(1, m + 1):
            # Check if a path exists from this ending position to reach the maximum LCS
            if dp[end_i][n] == lcs_length:
                # We found the earliest valid ending position that gives the max LCS
                earliest_end = end_i
                break
        
        i, j = (earliest_end, n) if earliest_end is not None else (m, n)
    else:
        # Standard backtracking from the bottom-right
        i, j = m, n

    positions = []

    # Backtrack to find the positions of the LCS in str1
    while i > 0 and j > 0:
        if reference_words[i - 1] == candidate_words[j - 1]:
            positions.append(i - 1)
            i -= 1
            j -= 1
        # If the LCS continues in the up direction, move up
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        # Otherwise, move left
        else:
            j -= 1
    
    positions.reverse()
    start_pos = positions[0]
    end_pos = positions[-1]

    # Find optimal start position and end position in the LCS by maximizing the RougeL score
    if is_find_optimal_lcs:
        # Optimal start position according to RougeL score
        start_scores = [
            calculate_rouge_L_score(
                reference_words[pos:positions[-1] + 1],
                candidate_words,
                lcs_length - i
            )
            for i, pos in enumerate(positions)
        ]
        start_pos = positions[start_scores.index(max(start_scores))]
        start_pos_index = positions.index(start_pos)

        # Optimal end position according to RougeL score
        end_scores = []
        for i, pos in enumerate(positions):
            if i < start_pos_index:
                end_scores.append(0)  # Score is 0 for any position before the new start position
                continue
                
            score = calculate_rouge_L_score(
                reference_words[start_pos:pos + 1],
                candidate_words,
                i - start_pos_index + 1
            )
            end_scores.append(score)
            
        end_pos = positions[end_scores.index(max(end_scores))]
        end_pos_index = positions.index(end_pos)
        lcs_length = (lcs_length - start_pos_index) - ((lcs_length - start_pos_index) - end_pos_index)

    return lcs_length, start_pos, end_pos

def find_segment_with_longest_common_subsequence(quote, chunk):
    # Clean and tokenize for finding the LCS
    document_words = clean_text(chunk)
    quote_words = clean_text(quote)
    
    lcs_length, start_pos, end_pos = find_lcs_length(document_words, quote_words, is_find_optimal_lcs = True)
    
    if lcs_length > 0:
        # Track both words and their character positions in the original text
        original_words = []
        word_positions = []
        current_pos = 0
        
        # Split while preserving positions
        while current_pos < len(chunk):
            # Skip leading whitespace
            while current_pos < len(chunk) and chunk[current_pos].isspace():
                current_pos += 1
            
            if current_pos >= len(chunk):
                break
                
            word_start = current_pos
            # Find end of word
            while current_pos < len(chunk) and not chunk[current_pos].isspace():
                current_pos += 1
            
            word = chunk[word_start:current_pos]
            if word:  # Only add non-empty words
                original_words.append(word)
                word_positions.append((word_start, current_pos))
        
        # Create mapping from clean words to original positions
        clean_to_original_map = {}
        current_clean_idx = 0
        
        for i, word in enumerate(original_words):
            cleaned_word = clean_text(word)
            if cleaned_word:  # If the word isn't empty after cleaning
                clean_to_original_map[current_clean_idx] = i
                current_clean_idx += 1
        
        # Get the original word indices
        original_start_idx = clean_to_original_map[start_pos]
        original_end_idx = clean_to_original_map[end_pos]
        
        # Get the character positions from our tracked positions
        segment_start = word_positions[original_start_idx][0]
        segment_end = word_positions[original_end_idx][1]
        
        # Extract the segment using character positions
        segment = chunk[segment_start:segment_end]
        
        # Calculate ROUGE-L score for the segment
        rouge_score = calculate_LCS_and_rouge_L_score(segment, quote)
        
        return segment, segment_start, segment_end, rouge_score
    return "", 0, 0, 0

def truncate_if_above_max_length(text, max_length):
    if len(text) <= max_length:
        return text
    
    words = text.split()
    final_text = ""
    
    # Add words one by one until we exceed max_length
    for word in words:
        if len(final_text + word + " ") > max_length:
            break
        final_text += word + " "
    
    return final_text.rstrip() + "..."

# Post processing that should always be applied when markdown is used.
def markdown_post_processing(text: str) -> str:
    # Replace the dollar signs to fix the markdown
    return text.replace("$", "\$")

def fix_links_and_emails(text):
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # URL pattern
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    
    # Add an empty space between the @ and the domain name
    def process_email(match):
        email = match.group(0)
        return email.replace('@', '&#8203;@')
    
    # Add an empty space between the . and the next character, same thing for //
    def process_url(match):
        url = match.group(0)
        return url.replace('.', '&#8203;.').replace('//', '&#8203;//')
    
    text = re.sub(email_pattern, process_email, text)
    text = re.sub(url_pattern, process_url, text)

    # Remove the square brackets in the markdown for links, plus any stars at the start and end
    # Ex : **[workforce analysis](url)** => workforce analysis (url)
    markdown_link_pattern = r"\**\[([^\]]+)\](\([^)]+\))\**"
    text = re.sub(markdown_link_pattern, r"\1 \2", text)

    return text

def create_segment_html(matched_segment, document_id, document_title, document_url):
    if "IPG" in document_id: # Sometimes, IPG is not at the start of the document_id
        segment_type = "ipg"
    else:
        # Determine segment type based on document_id prefix
        segment_type = document_id.split('-')[0].lower()

    segment_class = {
        'clc': 'segment-clc',
        'clsr': 'segment-clsr',
        'ipg': 'segment-ipg'
    }.get(segment_type, 'segment-default')

    segment_identifier = truncate_if_above_max_length(document_id + ": " + document_title, 60)

    # Fix all links and emails so that the html parser don't break the a tag below by converting them to html entities
    modified_segment = fix_links_and_emails(matched_segment)
    
    return f"""\n<a href="{document_url}" class="matched-segment {segment_class}"><span class="matched-segment-text">{modified_segment}</span><span class="matched-segment-divider"></span><span class="matched-segment-id">{segment_identifier}</span></a><br/>"""

def format_matched_segment_text(text):
    formatted_text = text

    # Remove any , at the start and end of the text
    formatted_text = formatted_text.strip(',;:" ')

    if formatted_text and formatted_text[0].islower():
        formatted_text = "..." + formatted_text
        
    if formatted_text and formatted_text[-1] not in '.!?':
        formatted_text = formatted_text + "..."
        
    return formatted_text

def format_matched_segment_text_headers(text):
    words = text.split()
    
    # Find the first non-capitalized word
    first_non_cap_index = 0
    for i, word in enumerate(words):
        if not word.isupper():
            first_non_cap_index = i
            break
    
    # Count uppercase words and 'I' words only at the start
    cap_word_count = first_non_cap_index
    i_word_count = sum(1 for word in words[:first_non_cap_index] if word == 'I')
            
    # If we found 2 or more capitalized words at start (excluding I)
    if cap_word_count - i_word_count >= 2:
        # Convert header words to title case, preserving 'I'
        header_words = ['I' if word == 'I' else word.title() for word in words[:cap_word_count]]
        # Join header with remaining text
        formatted_text = ' '.join(header_words) + ': ' + ' '.join(words[cap_word_count:])
        return formatted_text
        
    return text

def format_matched_segment(matched_segment, chunk_text, chunk_index, start_pos, end_pos, chunk_id, chunk_title, chunk_url, include_html, include_attribution=False, score=None, include_complete_sentence=False):
    # Optional : Remove the ... before and after the matched segment, and try to include the full sentence
    if include_complete_sentence:
        # Find the start of the sentence
        sentence_start = start_pos
        while sentence_start > 0:
            # Look for sentence ending punctuation followed by space/newline
            if chunk_text[sentence_start-1] in '.!?\n;:' or chunk_text[sentence_start].isupper():
                break
            sentence_start -= 1

        # Find the end of the sentence
        sentence_end = end_pos - 1 if end_pos > 0 else 0
        while sentence_end < len(chunk_text):
            if chunk_text[sentence_end].isupper():
                break
            if chunk_text[sentence_end] in '.!?\n;':
                sentence_end += 1
                break
            sentence_end += 1

        # Update matched_segment to include the full sentence
        matched_segment = chunk_text[sentence_start:sentence_end].strip()
    else:
        # Format the matched segment text
        matched_segment = format_matched_segment_text(matched_segment)
        sentence_start = start_pos
        sentence_end = end_pos

    matched_segment = format_matched_segment_text_headers(matched_segment)

    # Format the attribution string
    if not include_attribution:
        attribution = None
    elif score is not None:
        attribution = f'(chunk {chunk_index + 1}, position {sentence_start} to {sentence_end}, score {score:.3f})'
    else:
        attribution = f'(chunk {chunk_index + 1}, position {sentence_start} to {sentence_end})'

    # Create replacement text with the actual matched segment
    if include_html:
        replacement_text = create_segment_html(matched_segment, chunk_id, chunk_title, chunk_url)
    else:
        replacement_text = f'"{matched_segment}"'

    return replacement_text, attribution

# Verifies quotes in an LLM answer against source chunks and adds attribution.
def verify_and_attribute_quotes(chunks, llm_answer, threshold, include_html=False, include_attribution=False, include_complete_sentence=False):
    # Extract regular quotes sections
    quotes = re.findall(r'"([^"]*)"', llm_answer)
    
    quotes = [quote for quote in quotes if not quote.endswith('**')] # Remove any quotes that ends with **

    # Remove duplicates
    quotes = list(set(quotes))

    # Order them in ascending order of nb of words
        # Fixes the bug when replacing a smaller quote after a lrger one, the text within the larger quote ise repalced twice (creates a link in a link)
    quotes = sorted(quotes, key=lambda x: len(x.split()))

    modified_answer = llm_answer
    
    # Process each quote
    for quote in quotes:
        # Skip quotes with X or fewer non-capitalized words
        if len([word for word in quote.split() if not word.isupper()]) <= QuotationsConfig.min_non_header_words_in_quote:
            continue
            
        attribution = replacement_text = None
        best_score = 0
        best_chunk_index = -1
        best_chunk_text = best_match_info = best_chunk_id = best_chunk_title = best_chunk_url = None
        
        for chunk_index, (chunk_text, chunk_url, chunk_id, chunk_title) in enumerate(chunks):
            # Remove the title from the start of the quote (and only the start)
            quote_without_title = quote.replace(chunk_title, '', 1).strip()

            # Skip the chunk if the len of words in the quote that are not the title is <= min_non_header_words_in_quote
            if len([word for word in quote_without_title.split() if not word.isupper()]) <= QuotationsConfig.min_non_header_words_in_quote:
                continue

            segment, start, end, score = find_segment_with_longest_common_subsequence(quote, chunk_text)
            
            if score > best_score:
                best_score = score
                best_chunk_text = chunk_text
                best_match_info = (segment, start, end)
                best_chunk_index = chunk_index
                best_chunk_id = chunk_id
                best_chunk_title = chunk_title
                best_chunk_url = chunk_url

        # Only use the match if score is > threshold
        if best_score > threshold and best_match_info:
            matched_segment, start, end = best_match_info
            replacement_text, attribution = format_matched_segment(
                matched_segment, best_chunk_text, best_chunk_index, start, end,
                best_chunk_id, best_chunk_title, best_chunk_url, include_html, include_attribution, best_score, include_complete_sentence
            )
        else:
            attribution = "(no source found for the quote)"

        if replacement_text is None and include_attribution:
            replacement_text = f'"{quote}"'
        elif replacement_text is None:
            continue

        # Remove prefixed markdown *, quotes "" and any parentheses that follow
            # Also matches markdown * exact quote (even though such quotes are not longer searched for, see re.findall above)
        pattern = rf'-*\**\n?\s*(?:\*?\s*"{re.escape(quote)}"|\*\s*{re.escape(quote)})(\s*\([^)]*\)+)?\.?;?\**'
        modified_answer = re.sub(pattern, f'\n{replacement_text}{" " + attribution if attribution is not None else ""}', modified_answer)
        
    return modified_answer

if __name__ == "__main__":

    ids = ["CLC-196"]

    chunks = [("""ID: CLC-196, Title: DIVISION V-General Holidays: Holiday pay, Parents: Canada Labour Code / PART III-Standard Hours, Wages, Vacations and Holidays, Section 196, Text: Section 196 (1) Subject to subsections (2) and (4), an employer shall, for each general holiday, pay an employee holiday pay equal to at least one twentieth of the wages, excluding overtime pay, that the employee earned with the employer in the four-week period immediately preceding the week in which the general holiday occurs. Marginal note: Employees on commission (2) An employee whose wages are paid in whole or in part on a commission basis and who has completed at least 12 weeks of continuous employment with an employer shall, for each general holiday, be paid holiday pay equal to at least one sixtieth of the wages, excluding overtime pay, that they earned in the 12-week period immediately preceding the week in which the general holiday occurs. (3) <script>echo("test")</script>
              
    [Repealed, 2018, c. 27, s. 458] Marginal note: Continuous operation employee not reporting for work (4) An employee who is employed in a continuous operation is not entitled to holiday pay for a general holiday (a) on which they do not report for work after having been called to work on that day; or (b) for which they make themselves unavailable to work when the conditions of employment in the industrial establishment in which they are employed (i) require them to be available, or (ii) allow them to make themselves unavailable. (5) [Repealed, 2018, c. 27, s. 458] R.S., 1985, c. L-2, s. 196 2012, c. 31, s. 221 2018, c. 27, s. 458 Previous Version Marginal note: Additional pay for holiday work""", "https://laws-lois.justice.gc.ca/eng/acts/l-2/FullText.html#h-342421", "CLC-196", "Vacations and Holidays Previous Version Marginal note for your information")]

    llm_answer = """
        According to Section 196 of the Canada Labour Code, 
        **"Text: an employer shall, for each general holiday, pay an employee holiday pay equal to at least one twentieth of the wages\nexcluding overtime pay, that the employee earned with the employer in the four-week period immediately preceding the week in which the general holiday occurs in."**
    """

    start_time = time.time()
    result = verify_and_attribute_quotes(chunks, llm_answer, QuotationsConfig.threshold_rouge_score, True, False, False)
    print(result)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")