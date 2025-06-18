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

from rag_utils.db_config import EmbeddingModel, ModelsConfig, VectorDBDataFiles
from rag.extract_ipgs import extract_ipgs_main
from rag.extract_pdf import extract_pdfs_main
from rag.extract_law import extract_law_main
from rag.extract_page import extract_pages_main

if __name__ == "__main__":
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    selected_tokenizer = selected_model.model.tokenizer
    selected_token_limit = selected_tokenizer.model_max_length - 45 # Remove 45 tokens for the upper limit of the metadata included at the start of each embedding

    databases = VectorDBDataFiles.databases

    for db in databases:
        db_name = db["name"]
        save_html = db.get("save_html", False)
        print(f"Processing {db_name}...")

        os.makedirs(f"outputs/{db_name}", exist_ok=True)

        db_ipg = db.get("ipg")
        if db_ipg:
            extract_ipgs_main(db_ipg, db_name, save_html, selected_tokenizer, selected_token_limit)

        db_law = db.get("law")
        if db_law:
            extract_law_main(db_law, db_name, selected_tokenizer, selected_token_limit)

        db_pages = db.get("pages")
        if db_pages:
            extract_pages_main(db_pages, db_name, save_html, db.get("pages_blacklist"), selected_tokenizer, selected_token_limit)

        db_pdfs = db.get("pdfs")
        if db_pdfs:
            extract_pdfs_main(db_pdfs, db_name, selected_tokenizer, selected_token_limit)