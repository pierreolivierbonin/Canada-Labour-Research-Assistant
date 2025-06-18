from typing import Tuple, List
import os
import csv
from dataclasses import dataclass
import re
from urllib.parse import urlparse

@dataclass
class TextChunk:
    text: str
    headers: List[str] 
    subheaders: List[str]
    tag_id: str

@dataclass
class Page:
    id: str
    title: str
    url: str
    hierarchy: List[str]
    url_hierarchy: List[str]
    linked_pages: List[str]
    text: str
    chunks: List[TextChunk]
    date_modified: str
        
def get_base_url(url: str) -> str:
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.scheme else parsed_url.netloc
    return base_url

# Add hashtag markers around header text for easier parsing later
def add_header_tags(soup):
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        tag_name = tag.name
        text = tag.get_text(strip=True)
        id = tag["id"] if tag.get("id") else ""
        tag.string = f"#{tag_name}#{text}/{id}#{tag_name}#"

def extract_main_content(soup) -> Tuple[str, List[str]]:
    main_content = soup.find('main')

    if not main_content:
        print("Warning: No <main> tag found, using body instead")
        main_content = soup.find('body')

    if not main_content:
        print("Warning: No <main> or <body> tag found, skipping page")
        return "", []
    
    linked_pages = []

    # Extract links from the element
    for link in main_content.find_all('a'):
        href = link.get('href')
        if href and href.startswith('/') and href not in linked_pages:
            linked_pages.append(href)

    # Add header tags before content extraction
    add_header_tags(soup)

    text_content = main_content.get_text()
    
    return text_content, list(linked_pages)

def extract_reference_section_number(text: str) -> List[str]:
    # Find all occurrences of "Section(s) X" in the text, case insensitive
    section_pattern = r"sections? (\d+)"
    sections = re.findall(section_pattern, text, re.IGNORECASE)
    
    # Remove duplicates, and sort
    section_numbers = sorted(list(set(sections)))
    return section_numbers

# Loop over all subheaders and remove the last / and everything after it
def extract_id_from_headers(headers: List[str], current_tag_id: str, is_subheader: bool = False) -> Tuple[List[str], str]:
    tag_id = ""
    
    for i, header in enumerate(headers):
        # Only overwrite if it's the first subheader, or the latest main header (otherwise, might link to a subheader further down the page from the relevant chunk)
        if i == 0 or not is_subheader:
            tag_id = header.split("/")[-1]

        # Remove the last / and everything after it
        last_slash_position = header.rfind("/")
        if last_slash_position != -1:
            headers[i] = headers[i][:last_slash_position]

    if tag_id:
        current_tag_id = tag_id

    return headers, current_tag_id

# Remove the id from the header
def format_header(header: str) -> str:
    # Remove the last / and everything after it
    last_slash_position = header.rfind("/")
    if last_slash_position != -1:
        header = header[:last_slash_position]

    return header.upper()

def chunk_text(text: str, tokenizer, token_limit: int) -> List[TextChunk]:
    chunks = []
    current_text = ""
    current_token_count = 0
    current_headers = []
    current_subheaders = []
    current_tag_id = ""
    
    # Find all headers (h1 or h2) with their positions
    header_pattern = r'#h[12]#.*?#h[12]#'
    header_matches = list(re.finditer(header_pattern, text))
    
    # Split text into segments using header positions
    segments = []
    for i, match in enumerate(header_matches):
        start_pos = match.start()
        # If this is the last header, take all remaining text
        if i == len(header_matches) - 1:
            segments.append(text[start_pos:])
        else:
            # Otherwise take text until the next header
            next_start = header_matches[i + 1].start()
            segments.append(text[start_pos:next_start])

    # Insert empty main header at the start if no main headers found (only subheaders)
    if len(segments) == 0:
        text = "#h1##h1#" + text
        segments.append(text)
    
    # Process each segment
    for segment in segments:
        segment_title = ""
        header_match = re.search(r'#h[12]#(.*?)#h[12]#', segment)
        if header_match:
            segment_title = header_match.group(1)
        
        # Find all subsegment headers (h3/h4) with their positions
        subheader_pattern = r'#h[34]#.*?#h[34]#'
        subheader_matches = list(re.finditer(subheader_pattern, segment))
        
        # Split segment into subsegments using header positions
        subsegments = []
        if subheader_matches:
            for i, match in enumerate(subheader_matches):
                start_pos = match.start()
                # If this is the last subheader, take all remaining text
                if i == len(subheader_matches) - 1:
                    subsegments.append(segment[start_pos:])
                else:
                    # Otherwise take text until the next subheader
                    next_start = subheader_matches[i + 1].start()
                    subsegments.append(segment[start_pos:next_start])
        else:
            # If no subsegments found, use entire segment
            subsegments = [segment]
        
        # Process subsegments
        for subsegment in subsegments:
            subsegment_title = ""
            subsegment_header_match = re.search(r'#h[34]#(.*?)#h[34]#', subsegment)
            if subsegment_header_match:
                subsegment_title = subsegment_header_match.group(1)

            # Convert all headers to uppercase + remove hashtags and ids
            subsegment = re.sub(r'#h[1-4]#(.*?)#h[1-4]#', lambda m: format_header(m.group(1)), subsegment).strip()

            # Check if adding this subsegment would exceed token limit
            potential_text = current_text + '  ' + subsegment if current_text else subsegment
            subsegment_token_count = len(tokenizer.encode(subsegment))
            
            if current_token_count + subsegment_token_count < token_limit:
                # Add to current chunk
                current_text = potential_text
                if segment_title and segment_title not in current_headers:
                    current_headers.append(segment_title)
                if subsegment_title:
                    current_subheaders.append(subsegment_title)

                current_token_count += subsegment_token_count
            else:
                # Create new chunk
                if current_text: 
                    formatted_headers, current_tag_id = extract_id_from_headers(current_headers.copy(), current_tag_id)
                    formatted_subheaders, current_tag_id = extract_id_from_headers(current_subheaders.copy(), current_tag_id, is_subheader=True)

                    chunks.append(TextChunk(
                        text=current_text.replace("\n", " ").replace("\r", " "),
                        headers=formatted_headers,
                        subheaders=formatted_subheaders,
                        tag_id=current_tag_id
                    ))

                current_text = subsegment
                current_headers = [segment_title] if segment_title else []
                current_subheaders = [subsegment_title] if subsegment_title else []
                current_token_count = subsegment_token_count

    # Add final chunk
    if current_text:
        formatted_headers, current_tag_id = extract_id_from_headers(current_headers.copy(), current_tag_id)
        formatted_subheaders, current_tag_id = extract_id_from_headers(current_subheaders.copy(), current_tag_id, is_subheader=True)
        
        chunks.append(TextChunk(
            text=current_text.replace("\n", " ").replace("\r", " "),
            headers=formatted_headers,
            subheaders=formatted_subheaders,
            tag_id=current_tag_id
        ))
    
    return chunks

# Extract the date modified from the page
def extract_date_modified(soup) -> str:
    date_element = soup.find('time', property='dateModified')
    if date_element:
        return date_element.get_text(strip=True)
    return ""

def get_page_csv_row(page: Page) -> List[str]:
    reference_section_number = extract_reference_section_number(page.text)
    return [page.id, page.title, page.url, " / ".join(page.hierarchy), " / ".join(page.url_hierarchy), "|".join(page.linked_pages) if page.linked_pages else "", ";".join(reference_section_number) if reference_section_number else "", page.date_modified]

def save_to_csv(pages: List[Page], database_name: str, filename: str, lang: str, is_pdf: bool = False):
    os.makedirs("outputs", exist_ok=True)
    lang_suffix = "_fr" if lang != "en" else ""
    csv_path = f"outputs/{database_name}/{filename}{lang_suffix}.csv"
    existing_page_ids = []
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'title', 'hyperlink', 'hierarchy', 'url_hierarchy', 'linked_pages', 'reference_section_number', 'date_modified'])
        for page in pages:
            if page.id in existing_page_ids:
                # Throw an error, not supposed to happen
                raise ValueError(f"Page {page.id} already exists in {csv_path}")
            
            writer.writerow(get_page_csv_row(page))
            existing_page_ids.append(page.id)

    csv_path = f"outputs/{database_name}/{filename}{lang_suffix}_chunks.csv"
    total_chunks = 0

    # Add new code to write chunks CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'page_id', 'headers', 'subheaders', 'hyperlink', 'text'])
        writer.writeheader()
        
        # Go through all pages and their chunks
        for page in pages:
            chunk_counter = 1
            
            for chunk in page.chunks:
                chunk_id = f"{page.id}_{chunk_counter}"
                tag_prefix = "page=" if is_pdf else "" # tag = page on pdfs
                writer.writerow({
                    'id': chunk_id,
                    'page_id': page.id,
                    'headers': chunk.headers,
                    'subheaders': chunk.subheaders,
                    'hyperlink': page.url + "#" + tag_prefix + chunk.tag_id,
                    'text': chunk.text
                })
                chunk_counter += 1
                total_chunks += 1

    print(f"Saved {len(pages)} pages to {csv_path} with {total_chunks} chunks")
