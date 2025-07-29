#!/usr/bin/env bash

echo "Extracting data for databases..."
python ./scripts/extract_for_database.py

echo "Creating database with embeddings..."
python ./scripts/create_database_with_specific_embeddings.py

echo "Database creation/update completed successfully!"