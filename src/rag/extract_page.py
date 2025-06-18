#!/usr/bin/env python3
"""
This script extracts content from Canada government pages and saves them to a CSV file.
For each page it:
  - Creates a Page object containing the page's metadata and content
  - Extracts the navigation hierarchy from the header
  - Extracts the main content text
  - Saves the page data to a CSV if it doesn't already exist
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

from rag_utils.page_utils import Page, extract_date_modified, chunk_text, extract_main_content, save_to_csv, get_base_url

MAX_BATCH_SIZE = 10
#BASE_URL = "https://www.canada.ca"

PROCESSED_LINKS = set()
USED_TITLES = set()

def extract_hierarchy(soup) -> Tuple[List[str], List[str]]:
    hierarchy = []
    url_hierarchy = []
    breadcrumb = soup.find('ol', class_='breadcrumb')
    if breadcrumb:
        for item in breadcrumb.find_all('li'):
            text = item.get_text(strip=True)
            link = item.find('a')
            if text:
                hierarchy.append(text)
            if link and link.get('href'):
                url_hierarchy.append(link.get('href'))
    return hierarchy, url_hierarchy

# Extract the title from the page
def extract_title(soup) -> str:
    h1 = soup.find('h1')
    if h1:
        return h1.get_text(strip=True)
    return ""

# Extract links from table of contents (Steps)
def extract_toc_links(soup, base_url) -> List[str]:
    toc = soup.find('ul', class_='toc')
    if not toc:
        return []
    
    links = []
    for link in toc.find_all('a'):
        href = link.get('href')
        if href and href.startswith('/'):
            links.append(f"{base_url}{href}")
    return links

# Save HTML content to a file in the outputs/canada_html directory using the page title
def save_html_content(content: str, title: str, current_language: str, database_name: str):
    language_suffix = "_fr" if current_language != "en" else ""
    output_dir = os.path.join("outputs", database_name, "pages_html" + language_suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the title for use as filename
    clean_title = title.strip().replace("/", "_").replace("\\", "_")
    clean_title = "".join(c for c in clean_title if c.isalnum() or c in "_ -")
    
    # Handle duplicate titles
    if clean_title in USED_TITLES:
        print(f"Duplicate title: {clean_title}, overwriting previous file")
    else:
        USED_TITLES.add(clean_title)
    
    # Save content
    filepath = os.path.join(output_dir, f"{clean_title}{language_suffix}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def process_page(url: str, current_depth: int, tokenizer, token_limit: int, current_language: str, database_name: str, save_html: bool, blacklist_urls: List[str], skip_toc: bool = False):
    current_processed_pages = []
    print(f"Processing {url} at depth {current_depth}")
    base_url = get_base_url(url)
    
    try:     
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = extract_title(soup)
        
        # Check for table of contents if not skipping
        if save_html:
            # Save HTML content with title
            save_html_content(response.text, title, current_language, database_name)
        
        # Check for table of contents if not skipping
        if not skip_toc:
            toc_links = extract_toc_links(soup, base_url)
            if toc_links:
                print(f"Found table of contents in {url}, processing sub-pages...")
                for full_url in toc_links:
                    if full_url not in PROCESSED_LINKS:
                        PROCESSED_LINKS.add(full_url) # Mark as processed
                        processed_pages = process_page(full_url, current_depth, tokenizer, token_limit, current_language, database_name, save_html, blacklist_urls, skip_toc=True)
                        current_processed_pages.extend(processed_pages)
                return current_processed_pages
        
        # Extract page components
        hierarchy, url_hierarchy = extract_hierarchy(soup)
        date_modified = extract_date_modified(soup)

        # Extract main content
        text, linked_pages = extract_main_content(soup)

        if not text:
            print(f"Warning: No text found for {url}, skipping chunking")
            return current_processed_pages
        
        # Chunk the text
        chunks = chunk_text(text, tokenizer, token_limit)
        
        page = Page(None, title, url, hierarchy, url_hierarchy, linked_pages, text, chunks, date_modified)
        current_processed_pages.append(page)
        
        # Process linked pages if depth allows
        if current_depth < 1:
            print(f"Processing links from {url} at depth {current_depth}")
            sub_links_to_process = []
                
            # Mark all links as processed before starting
            for link in linked_pages:
                full_url = f"{base_url}{link}"
                if full_url not in PROCESSED_LINKS and not any(link.startswith(root_url) for root_url in blacklist_urls):
                    PROCESSED_LINKS.add(full_url)
                    sub_links_to_process.append(full_url)
            
            # Process in parallel only at depth 0
            if current_depth == 0:
                # Process in batches of 10
                with ThreadPoolExecutor(max_workers=MAX_BATCH_SIZE) as executor:
                    futures = [executor.submit(process_page, url, current_depth + 1, tokenizer, token_limit, current_language, database_name, save_html, blacklist_urls) for url in sub_links_to_process]

                    # Wait for batch completion and add results in order
                    # for future in sorted(futures.keys(), key=lambda f: futures[f]):
                    for future in futures:
                        processed_pages = future.result()
                        if processed_pages:
                            current_processed_pages.extend(processed_pages)
            else:
                # Process sequentially for depth > 0
                for link in sub_links_to_process:
                    processed_pages = process_page(link, current_depth + 1, tokenizer, token_limit, current_language, database_name, save_html, blacklist_urls)
                    current_processed_pages.extend(processed_pages)

    except Exception as e:
        print(f"Error processing {url}: {e}")

    return current_processed_pages

def extract_pages_main(pages_dict, database_name, save_html, blacklist_dict, selected_tokenizer, selected_token_limit):
    #languages = ["en", "fr"]

    global PROCESSED_LINKS

    for language in pages_dict:
        print(f"Processing pages in {language}...")

        #pages_to_process = WebCrawlConfig.canada_pages_ids_and_urls if language == "en" else WebCrawlConfig.canada_pages_ids_and_urls_fr
        pages_to_process = pages_dict[language]
        
        # Initialize PROCESSED_LINKS with starting pages
        PROCESSED_LINKS = set(pages_to_process)

        # BLACKLIST_ROOT_URLS = set(WebCrawlConfig.canada_pages_blacklist_urls if language == "en" else WebCrawlConfig.canada_pages_blacklist_urls_fr)
        blacklist_urls = blacklist_dict[language] if blacklist_dict else []
        #BLACKLIST_ROOT_URLS = set(blacklist_urls)

        all_processed_pages = []
        
        for id_prefix, page_url in pages_to_process:
            processed_pages = process_page(page_url, 0, selected_tokenizer, selected_token_limit, language, database_name, save_html, blacklist_urls)

            # Set the id for each page (Otherwise, might not be in order due to parallel processing)
            for idx, page in enumerate(processed_pages):
                page.id = f"{id_prefix}-{idx + 1}"

            all_processed_pages.extend(processed_pages)

        save_to_csv(all_processed_pages, database_name, "pages", language)

if __name__ == "__main__":
    from rag_utils.db_config import EmbeddingModel, ModelsConfig, VectorDBDataFiles

    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    selected_tokenizer = selected_model.model.tokenizer
    selected_token_limit = selected_tokenizer.model_max_length - 45 # Remove 45 tokens for the upper limit of the metadata included at the start of each embedding

    databases = VectorDBDataFiles.databases

    for db in databases:
        db_name = db["name"]
        save_html = db.get("save_html", False)

        pages = db.get("pages")

        if pages:
            extract_pages_main(pages, db_name, save_html, db.get("pages_blacklist"), selected_tokenizer, selected_token_limit)