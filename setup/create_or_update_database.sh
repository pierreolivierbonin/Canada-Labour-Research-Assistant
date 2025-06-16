#!/usr/bin/env bash
python ./scripts/extract_canada_page.py &&
python ./scripts/extract_ipgs.py &&
python ./scripts/extract_toc.py &&
python ./scripts/extract_pdf.py &&
python ./scripts/create_database_with_specific_embeddings.py