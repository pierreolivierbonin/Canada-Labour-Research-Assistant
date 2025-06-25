#!/usr/bin/env bash
python ./scripts/extract_for_database.py &&
python ./scripts/create_database_with_specific_embeddings.py