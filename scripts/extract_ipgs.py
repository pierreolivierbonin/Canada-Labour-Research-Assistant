#!/usr/bin/env python3
"""
This script extracts Interpretations, Policies and Guidelines (IPGs) from the Canada Labour Program website.
For each IPG it:
  - Creates a Page object containing the IPG's metadata and content
  - Extracts content from the linked page
  - Saves all IPG data to a CSV file
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from dataclasses import dataclass
import os

from rag_utils.page_utils import Page, extract_date_modified, extract_main_content, save_to_csv, chunk_text
from rag_utils.db_config import EmbeddingModel, ModelsConfig, WebCrawlConfig

MAX_WORKERS = 10
BASE_URL = "https://www.canada.ca"

PROCESSED_IPG_IDS = []

@dataclass
class IPG:
    title: str
    url: str
    id: str
    table_title: str

def process_ipg_page(ipg: IPG, tokenizer, token_limit, current_language) -> Optional[Page]:
    try:
        full_url = urljoin(BASE_URL, ipg.url)
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        language_suffix = "_fr" if current_language != "en" else ""
        
        output_dir = f"outputs/ipgs_html{language_suffix}"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{ipg.id}{language_suffix}.html"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text, linked_pages = extract_main_content(soup)

        # Extract date modified
        date_modified = extract_date_modified(soup)

        # Add text chunking
        chunks = chunk_text(text, tokenizer, token_limit)

        print(f"Processed IPG: {ipg.title} - {full_url} (Hierarchy: {ipg.table_title})")
        print(f"Saved HTML to: {filename}")
        
        return Page(
            ipg.id,
            ipg.title + " - " + ipg.id,
            full_url,
            [ipg.table_title],
            [],
            linked_pages,
            text,
            chunks,
            date_modified
        )
    
    except Exception as e:
        print(f"Error processing {ipg.url}: {e}")
        return None

def extract_ipgs_from_table(table) -> List[IPG]:
    ipgs = []
    
    # Find table title (usually in a caption or preceding h2/h3)
    table_title = ""
    caption = table.find('caption')
    if caption:
        table_title = caption.get_text(strip=True)
    else:
        # Look for preceding header
        prev_elem = table.find_previous(['h2', 'h3'])
        if prev_elem:
            table_title = prev_elem.get_text()
    
    # Find the header row to determine column positions
    headers = table.find_all('th')
    title_idx = next((i for i, h in enumerate(headers) if 'Title' in h.get_text()), 0)
    number_idx = next((i for i, h in enumerate(headers) if 'Number' in h.get_text() or 'No.' in h.get_text()), 1)
    
    # Process each row
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        title_cell = cells[title_idx]
        number_cell = cells[number_idx]
        
        # Extract title and link
        link = title_cell.find('a')
        if not link:
            continue
            
        title = title_cell.get_text(strip=True)
        url = link.get('href')
        ipg_id = number_cell.get_text(strip=True)

        if ipg_id in PROCESSED_IPG_IDS:
            print(f"Skipping duplicate IPG: {ipg_id}")
            continue
        
        if url and title and ipg_id:
            ipgs.append(IPG(title, url, ipg_id, table_title))
            PROCESSED_IPG_IDS.append(ipg_id)
    
    return ipgs

def main():
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    selected_tokenizer = selected_model.model.tokenizer
    selected_token_limit = selected_tokenizer.model_max_length - 45 # Remove 45 tokens for the upper limit of the metadata included at the start of each embedding

    languages = ["en", "fr"]

    for language in languages:
        print(f"Processing IPGs in {language}...")

        global PROCESSED_IPG_IDS
        PROCESSED_IPG_IDS = []
        
        # Fetch main IPG page
        response = requests.get(WebCrawlConfig.ipg_url if language == "en" else WebCrawlConfig.ipg_url_fr)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        
        # Extract IPGs from all tables
        all_ipgs = []
        for table in tables:
            all_ipgs.extend(extract_ipgs_from_table(table))
        
        print(f"Found {len(all_ipgs)} IPGs to process")
        
        # Process IPG pages in parallel
        processed_pages = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ipg = {
                executor.submit(process_ipg_page, ipg, selected_tokenizer, selected_token_limit, language): ipg 
                for ipg in all_ipgs
            }
            
            for future in future_to_ipg:
                page = future.result()
                if page:
                    processed_pages.append(page)
        
        save_to_csv(processed_pages, "ipgs", language)

if __name__ == "__main__":
    main()
