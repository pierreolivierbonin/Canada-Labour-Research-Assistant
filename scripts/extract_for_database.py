#!/usr/bin/env python3
"""
This script extracts content from urls and saves them to CSV files.
It can extract from:
- IPGs
- Laws
- Pages (Mostly Canada.ca pages.)
- PDFs
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
from db_config import VectorDBDataFiles
from rag.rag_extractor import RagExtractor

if __name__ == "__main__":
    databases = VectorDBDataFiles.databases
    extractor = RagExtractor()

    for db in databases:
        db_name = db["name"]
        save_html = db.get("save_html", False)
        print(f"Processing {db_name}...")

        os.makedirs(f"outputs/{db_name}", exist_ok=True)

        db_ipg = db.get("ipg")
        if db_ipg:
            extractor.extract("ipg", db_ipg, db_name, save_html)

        db_law = db.get("law")
        if db_law:
            extractor.extract("law", db_law, db_name)

        db_pages = db.get("page")
        if db_pages:
            extractor.extract("page", db_pages, db_name, save_html, db.get("page_blacklist"))

        db_pdfs = db.get("pdf")
        if db_pdfs:
            extractor.extract("pdf", db_pdfs, db_name)