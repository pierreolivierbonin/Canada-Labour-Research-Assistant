"""
Extract Transport Canada Acts and Regulations Links

This script scrapes Transport Canada's website to collect links to all transport acts and regulations,
then generates a database collection configuration file for the Canada Labour Research Assistant.

What it does:
1. Scrapes https://tc.canada.ca/en/corporate-services/acts-regulations/list-regulations
2. Scrapes https://tc.canada.ca/en/corporate-services/acts-regulations/list-acts  
3. Filters and processes the links to only include valid law pages
4. Creates collections/transport_act_reg.json with the structured data

Usage:
1. Uncomment the line that calls this script in setup/create_or_update_database.sh
2. Run the database creation script:
   ./setup/create_or_update_database.sh

Alternatively, run manually:
  python scripts/extract_links_transport_acts_and_reg.py
  ./setup/create_or_update_database.sh
"""

import requests
import os
import json
from rag.page_utils import safe_beautifulsoup

# Process href by normalizing protocol, adding base URL if needed, and removing unwanted suffixes.
def process_href(href, base_url="https://tc.canada.ca"):
    if not href:
        return None
    
    # Replace http with https
    if href.startswith('http://'):
        href = href.replace('http://', 'https://')
    elif not href.startswith('https://'):
        href = base_url + href
    
    # Remove / at the end, if any
    href = href.rstrip('/')
    
    # Remove /page-1.html or /index.html from the end
    if href.endswith('/page-1.html'):
        href = href[:-len('/page-1.html')]
    elif href.endswith('/index.html'):
        href = href[:-len('/index.html')]
    elif href.endswith('/FullText.html'):
        href = href[:-len('/FullText.html')]
    
    return href

# Download from https://tc.canada.ca/en/corporate-services/acts-regulations/list-regulations
url = "https://tc.canada.ca/en/corporate-services/acts-regulations/list-regulations"
response = requests.get(url)
response.raise_for_status()

with safe_beautifulsoup(response.content) as soup:
    # Get only the first div with the specified class
    div = soup.find('div', class_='block-field-blocknodetcpagebody')
    links_all_transport_regulations = []

    if not div:
        print("No div found with class 'block-field-blocknodetcpagebody'")
        exit()

    # Loop over all ul elements within the div
    for ul in div.find_all('ul'):
        # Loop over all a elements within each ul
        for a in ul.find_all('a', href=True):
            href = process_href(a.get('href'))
            if href:
                links_all_transport_regulations.append(href)

# Extract acts from the acts page
acts_url = "https://tc.canada.ca/en/corporate-services/acts-regulations/list-acts"
acts_response = requests.get(acts_url)
acts_response.raise_for_status()

with safe_beautifulsoup(acts_response.content) as acts_soup:
    links_all_transport_acts = []

    tbody = acts_soup.find('tbody')

    if not tbody:
        print("No tbody found")
        exit()
    
    # Look for the tbody tag and extract all links from them
    for a in tbody.find_all('a', href=True):
        href = process_href(a.get('href'))
        if href and "laws-lois.justice.gc.ca/eng/acts/L-2" not in href:
            links_all_transport_acts.append(href)

# Create a folder for the links
os.makedirs('extracted_data', exist_ok=True)

# create a set of the links to remove duplicates
links_all_transport_regulations = list(set(links_all_transport_regulations))
links_all_transport_acts = list(set(links_all_transport_acts))

# Combine all links from both acts and regulations
all_links = links_all_transport_acts + links_all_transport_regulations

# Save the regulations to a file
with open('extracted_data/all_transport_regulations_links.txt', 'w', encoding='utf-8') as f:
    f.writelines(link + '\n' for link in links_all_transport_regulations)

# Save the acts to a file
with open('extracted_data/all_transport_acts_links.txt', 'w', encoding='utf-8') as f:
    f.writelines(link + '\n' for link in links_all_transport_acts)

# Create a combined file with acts first, then regulations
with open('extracted_data/all_transport_acts_and_regulations_links.txt', 'w', encoding='utf-8') as f:
    f.writelines(link + '\n' for link in all_links)

# Create transport_act_reg database configuration
transport_act_reg = {
    "name": "transport_act_reg",
    "languages": ["en"],
    "ressource_name": {
        "en": "Transport Acts & Regulations"
    },
    "law": {
        "en": []
    }
}

law_pages = [
    "https://laws-lois.justice.gc.ca",
    "https://lois-laws.justice.gc.ca",
    "https://laws.justice.gc.ca"
]

not_included_pages = []

# Process each link and categorize
for link in all_links:
    link_id_tuple = None
    
    # Extract ID from the last part of the URL for page links too
    link_id = link.split('/')[-1]
    if link_id == "royal-assent":
        link_id = link.split('/')[-2] # Get the previous part of the url
    link_id_tuple = (link_id, link)

    # Check if link starts with laws-lois.justice.gc.ca or lois-laws.justice.gc.ca
    if any(link.startswith(page) for page in law_pages):
        transport_act_reg["law"]["en"].append(link_id_tuple)
    else:
        not_included_pages.append(link)

# Save the transport_act_reg object as JSON
with open('collections/transport_act_reg.json', 'w', encoding='utf-8') as f:
    json.dump(transport_act_reg, f, indent=2, ensure_ascii=False)

if len(not_included_pages) > 0:
    with open('extracted_data/not_included_pages.txt', 'w', encoding='utf-8') as f:
        f.writelines(link + '\n' for link in not_included_pages)

print("--------------------------------")
print(f"Created transport_act_reg database configuration with:")
print(f"  - {len(transport_act_reg['law']['en'])} law links")
print(f"  - Saved to extracted_data/transport_act_reg.json")

if len(not_included_pages) > 0:
    print("--------------------------------")
    print("Some pages were not valid law pages, the correct links should be added manually if it can be found.")
    print(f"  - {len(not_included_pages)} pages not included")
    print(f"  - Saved to extracted_data/not_included_pages.txt")
    