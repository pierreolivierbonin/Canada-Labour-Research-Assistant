#!/usr/bin/env bash
# python ./scripts/extract_links_transport_acts_and_reg.py &&                                # uncomment to perform webcrawling for the first time
python ./scripts/extract_for_database.py --exclude "labour" "equity" "clement_use_case" &&                   # remove the "exclude" args to process everything
python ./scripts/create_database_with_specific_embeddings.py --exclude "labour" "equity" "clement_use_case" # embeds again the collections if files exist