#!/usr/bin/env bash
#python ./scripts/extract_links_transport_acts_and_reg.py && # UNCOMMENT TO CREATE TRANSPORT DATABASE
python ./scripts/extract_for_database.py &&
python ./scripts/create_database_with_specific_embeddings.py