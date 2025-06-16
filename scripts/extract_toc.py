#!/usr/bin/env python3
"""
This script scrapes the Canada Labour Code
It processes the table-of-contents recursively so that only the leaf sections are output.
For each leaf it:
  - Extracts the link URL, title, and (if available) a section number.
  - Constructs a "Hierarchy" string from its parent nodes (skipping the topmost "Canada Labour Code").
  - Downloads the associated page and (if the URL contains an anchor) extracts only the text
    between the start of the corresponding <hX> tag and the next <hX>, if any.
  - Within each TOC entry, identifies individual sections by their section labels.
It writes each section's title, section number, hierarchy, URL and text as a row in csv files.
"""

import csv
import os
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
from typing import List, Optional
from rag_utils.db_config import EmbeddingModel, ModelsConfig, WebCrawlConfig

class SectionItem:
    def __init__(self, section_number: str, title: str, text: str, tag_id: str):
        self.section_number = section_number
        self.title = title
        self.text = text
        self.tag_id = tag_id

class TocItem:
    def __init__(self, title: str, section_number: str, link_url: str, hierarchy: str):
        self.title = title
        self.section_number = section_number
        self.link_url = link_url
        self.hierarchy = hierarchy
        self.sections: List[SectionItem] = []

# Recursively process a TOC element.
def parse_toc_items(ul, current_hierarchy, base_url) -> list[TocItem]:
    items = []
    for li in ul.find_all("li", recursive=False):
        # Get the first link in this <li>
        a = li.find("a", recursive=False)
        if not a:
            continue
        link_url = a.get("href", "")

        title = a.get_text(strip=True)
        
        # Extract section number if available from a <span class="sectionRange"> within the <li>.
        section_number = ""
        span_tag = li.find("span", class_="sectionRange")

        if span_tag:
            span_text = span_tag.get_text(strip=True)
            section_number = span_text.split("-")[0].strip().replace(".", "-")

        # Check if this <li> has children
        child_ul = li.find("ul", recursive=False)
        if child_ul:
            new_hierarchy = list(current_hierarchy)
            new_hierarchy.append(title)

            child_items = parse_toc_items(child_ul, new_hierarchy, base_url)
            items.extend(child_items)
        else:
            hierarchy_str = " / ".join(current_hierarchy)

            # Extract everything after the first #
            link_url = "#" + link_url.split("#")[1] if "#" in link_url else link_url

            items.append(TocItem(title, section_number, link_url, hierarchy_str))
    return items

# Fetch the main Labour Code page and parse the table-of-contents recursively.
def get_main_toc_links(base_url: str) -> list[TocItem]:
    response = requests.get(base_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    toc = soup.find('ul', class_='TocIndent')
    toc_items = parse_toc_items(toc, [], base_url)
    return toc_items

def extract_page_text(soup, url, is_schedule = False) -> Optional[List[SectionItem]]:
    parsed_url = urlparse(url)
    if parsed_url.fragment:
        header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

        # Look for the section header with the matching fragment id
        section_header = soup.find(header_tags, id=parsed_url.fragment)
        if not section_header:
            print(f"Section with ID '{parsed_url.fragment}' not found in {url}.")
            return None
        
        # Check if the header is within a <header> tag
        parent_header = section_header.find_parent('header')
        if parent_header:
            section_header = parent_header
            
        # Find all section and schedule labels within this section's content
        section_labels = []
        current = section_header
        while current:
            if isinstance(current, Tag):
                if current.name in header_tags and current != section_header:
                    break
                # Find any section or schedule labels in this tag
                label_class = "sectionLabel" if not is_schedule else "scheduleLabel"
                labels = current.find_all('span', class_=label_class)
                section_labels.extend(labels)
            current = current.next_sibling
            
        # Process each section label to create SectionItems
        sections = []
        for label in section_labels:
            # Get the section number from the label
            section_number = label.get_text(strip=True).replace("[", "").replace("]", "").replace("Section ", "").strip()
            
            # Find the parent container
            section_container_class = "Section" if not is_schedule else "Schedule"
            section_container_parent_tags = ['ul', 'p'] if not is_schedule else ['div']
            section_container = label.find_parent(section_container_parent_tags, class_=section_container_class)
            if not section_container:
                continue

            section_container_id = section_container.get('id')
                
            # Find the title - it's in a <p class="MarginalNote"> before the section container
            title = ""
            prev_sibling = section_container.previous_sibling
            while prev_sibling:
                if isinstance(prev_sibling, Tag):
                    # Stop if we hit a header tag
                    if prev_sibling.name in header_tags:
                        break
                    
                    if prev_sibling.name == 'p' and 'MarginalNote' in prev_sibling.get('class', []):
                        # Remove the wb-invisible span before getting text
                        wb_invisible = prev_sibling.find('span', class_='wb-invisible')
                        if wb_invisible:
                            wb_invisible.decompose()
                        title = prev_sibling.get_text(strip=True)
                        break
                prev_sibling = prev_sibling.previous_sibling
            
            # Extract text from this section until the next section or header
            section_text = ""
            current = section_container
            
            while current:
                if isinstance(current, Tag):
                    # Stop if we hit a new section, header, or <section> tag
                    if ('Section' in current.get('class', []) and current != section_container) or \
                       current.name in header_tags or current.name == 'section':
                        break
                    
                    current_dt = current.find('dfn')
                    if current_dt:
                        current_dt.decompose()
                    
                    section_text += "\n" + current.get_text(separator="\n", strip=True)
                current = current.next_sibling

            # Remove "Previous Version" from the end of the text (only look from the end)
            section_text = section_text.rsplit("Previous Version", 1)[0].strip() if section_text.endswith("Previous Version") else section_text
            section_text = section_text.replace("\n", " ").replace("\r", " ")
            
            sections.append(SectionItem(
                section_number=section_number.replace(" ", "_") if section_number else "",
                title=title,
                text=section_text.strip(),
                tag_id=section_container_id
            ))
        
        return sections
    
    return None

def process_toc_page(toc_url, file_name, tokenizer, token_limit, current_language):
    print("Fetching table of contents links...")
    toc_items = get_main_toc_links(toc_url)
    print(f"Found {len(toc_items)} leaf links.")
    soup = None

    full_text_name = "FullText" if current_language == "en" else "TexteComplet"
    full_page_url = f"{toc_url}{full_text_name}.html"

    try:
        response = requests.get(full_page_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
    except Exception as e:
        print(f"Error extracting text from {full_page_url}: {e}")
        return ""
    
    # Add markers around section numbers
    for section_label in soup.find_all('span', class_='sectionLabel'):
        section_label.string = f"Section {section_label.get_text(strip=True)}"

    language_suffix = "_fr" if current_language != "en" else ""
    
    # Open the CSV file for writing
    with open(f"outputs/{file_name}{language_suffix}.csv", "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["id", "title", "section_number", "hierarchy", "hyperlink", "text"])

        for toc_item in toc_items:
            url = requests.compat.urljoin(full_page_url, toc_item.link_url)
            print(f"Processing: {toc_item.title} - {url}")

            schedule_name = "SCHEDULE" if current_language == "en" else "ANNEXE"
            is_schedule = schedule_name in toc_item.title
            sections = extract_page_text(soup, url, is_schedule)
            if not sections:
                print(f"No sections found for {url}")
                continue

            id_prefix = file_name.upper() + "-"
            
            # Group sections by their main section number
            section_groups = {}
            for section in sections:
                # Get main section number (before the dot)
                main_section = section.section_number.split('.')[0]
                
                if main_section not in section_groups:
                    section_groups[main_section] = []
                section_groups[main_section].append(section)
            
            # Process each group of sections
            for main_section, group in section_groups.items():
                combined_sections = []
                current_combined = group[0]
                current_text = current_combined.text
                
                for next_section in group[1:]:
                    # Check if combining would exceed token limit
                    combined_text = f"{current_text} {next_section.text}"
                    token_count = len(tokenizer.encode(combined_text))
                    
                    if token_count < token_limit:
                        # Combine sections
                        current_text = combined_text
                        current_combined.text = current_text
                        current_combined.title = f"{current_combined.title} | {next_section.title}" if current_combined.title else next_section.title
                    else:
                        # Start new combined section
                        combined_sections.append(current_combined)
                        current_combined = next_section
                        current_text = current_combined.text
                
                # Add the last combined section
                combined_sections.append(current_combined)
                
                # Write combined sections to CSV
                for section in combined_sections:
                    id_text = f"{id_prefix}{section.section_number}"
                    title = toc_item.title + (": " + section.title if section.title and section.title != toc_item.title else "")
                    section_url = requests.compat.urljoin(full_page_url, "#" + section.tag_id)
                    
                    csv_writer.writerow([
                        id_text,
                        title,
                        section.section_number,
                        toc_item.hierarchy,
                        section_url,
                        section.text
                    ])

if __name__ == "__main__":
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    selected_tokenizer = selected_model.model.tokenizer
    selected_token_limit = selected_tokenizer.model_max_length - 45 # Remove 45 tokens for the upper limit of the metadata included at the start of each embedding

    languages = ["en", "fr"]

    for language in languages:
        print(f"Processing TOC in {language}...")
        
        documents = WebCrawlConfig.toc_filenames_and_urls if language == "en" else WebCrawlConfig.toc_filenames_and_urls_fr

        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        for file_name, toc_url in documents:
            process_toc_page(toc_url, file_name, selected_tokenizer, selected_token_limit, language)