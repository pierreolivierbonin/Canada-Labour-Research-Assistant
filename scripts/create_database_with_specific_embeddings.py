import csv
from dataclasses import fields
import os

import chromadb
import pandas as pd
from time import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
from config import ChromaDBSettings
from rag_utils.db_config import EmbeddingModel, VectorDBDataFiles

def main(collection_name:str,
         db_path:str,
         data_files_tuples:list[tuple[str, bool]],
         model:EmbeddingModel,
         distance_func:str,
         current_language:str
         ):
    

    # create client
    client = chromadb.PersistentClient(path=db_path)

    # set a custom max_seq_length
    model.max_seq_length = model.used_seq_length 

    # fetch or create collection
    collection = client.get_or_create_collection(name=collection_name,
                                                    embedding_function=model.model_chroma_callable,
                                                    metadata={
                                                    "hnsw:space":distance_func,
                                                })

    # Process CSV data and upsert it into the ChromaDB collection
    def process_and_upsert_data(collection, filename_arg, has_section_nb=True):

        language_suffix = "_fr" if current_language == "fr" else ""
        filename = filename_arg + language_suffix

        # Load and clean page data
        df = pd.read_csv(filename + ".csv", encoding='utf-8')
        df.fillna(value="N/A", inplace=True)
        
        if has_section_nb:
            # Common fields for both document types
            base_fields = {
                "Title": df.title.values,
                "Hierarchy": df.hierarchy.values,
                "Section": df.section_number.values,
                "Text": df.text.values
            }
            
            # Create augmented passages
            augmented_passages = [
                ", ".join([f'{k}: {v[i]}' for k, v in base_fields.items()])
                for i in range(len(df))
            ]
            
            # Common metadata fields
            base_metadata = {
                "id": df.id.values,
                "title": df.title.values,
                "hierarchy": df.hierarchy.values,
                "section_number": df.section_number.values,
                "main_section_number": df.section_number.apply(lambda x: x.split('.')[0] if isinstance(x, str) and '.' in x else x).values,
                "hyperlink": df.hyperlink.values
            }
            
            metadatas = [
                {k: v[i] for k, v in base_metadata.items()}
                for i in range(len(df))
            ]

            documents = list(df.text.values)
            ids = df["id"].values.tolist()

        else:
            # Process chunked pages
            # First create a dictionary of page metadata
            page_metadata = {}
            for _, row in df.iterrows():
                page_metadata[row['id']] = {
                    'title': row['title'],
                    'hierarchy': row['hierarchy'],
                    'hyperlink': row['hyperlink'],
                    'date_modified': row['date_modified']
                }
            
            # Now load and process chunks
            chunks_df = pd.read_csv(filename + "_chunks.csv", encoding='utf-8')
            chunks_df.fillna(value="N/A", inplace=True)
            
            # Create augmented passages for chunks
            augmented_passages = []
            metadatas = []
            documents = []
            ids = []
            
            for _, chunk in chunks_df.iterrows():
                page_id = chunk['page_id']
                page_meta = page_metadata[page_id]
                
                # Create augmented passage for chunk
                chunk_fields = {
                    "Title": page_meta['title'],
                    "Hierarchy": page_meta['hierarchy'],
                    "Text": chunk['text']
                }
                
                augmented_passages.append(
                    str([f"{k}: {v}" for k, v in chunk_fields.items()])
                )
                
                # Create metadata for chunk
                chunk_metadata = {
                    "id": page_id, # Store the page id (ex : underlying IPG id)
                    "title": page_meta['title'],
                    "hierarchy": page_meta['hierarchy'],
                    "hyperlink": chunk['hyperlink'],
                    "headers": chunk['headers'],
                    "subheaders": chunk['subheaders'],
                    "date_modified": page_meta['date_modified']
                }
                metadatas.append(chunk_metadata)
                
                documents.append(chunk['text'])
                ids.append(chunk['id'])

        # Generate embeddings and upsert to collection in batches
        embeddings = model.model_chroma_callable(augmented_passages)
        
        # Upsert to collection
        collection.upsert(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    # for _, file_info in db_dict.items():
    #     process_and_upsert_data(collection, *file_info)
    for file_path, has_section_nb in data_files_tuples:
        process_and_upsert_data(collection, file_path, has_section_nb)

    # quick check if the output makes sense
    queries = [
        "What are the rules applying to maternity leave?",
        "What does the notion of averaging of hours mean for federally regulated employers?",
        "What is constructive dismissal?",
        "What is the definition of danger?",
        "How to prevent harmful behaviour at work?"
    ]

    results = collection.query(
        query_texts=queries,        # Chroma will embed these for you
        n_results=3,                # how many results to return
        include=["metadatas", "distances", "embeddings"]
    )

    print(results.items()) # quick look at the results
    print(f"\n\nCollection created: {collection_name}")
    print("File 'collections.csv' already exists: ", os.path.exists(os.path.join(os.getcwd(), "collections.csv")))

    if not os.path.exists(os.path.join(os.getcwd(), "collections.csv")):
        with open("collections.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["collection_name", 
                             "embedding_model", 
                             "max_seq_length",
                             "used_seq_length",
                             "chroma_distance_func"])

    collection_already_exists = False
    needs_newline = False

    with open("collections.csv", 'rt') as f:
        content = f.read()
        if content and not content.endswith('\n'):
            needs_newline = True
        f.seek(0)  # Reset file pointer to beginning

        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if collection_name == row[0]: # if the username shall be on column 3 (-> index 2)
                collection_already_exists = True

    if collection_already_exists:
        print("Collection already exist. Records upserted but tracker 'collections.csv' remains untouched.")
    else:
        with open("collections.csv", 'a', newline='', encoding="utf-8") as f:
            if needs_newline:
                f.write('\n')
            writer = csv.writer(f)
            writer.writerow([collection_name, 
                        model.model_name,
                        model.max_seq_length,
                        model.used_seq_length,
                        distance_func])
            print("Collection name and specs saved to list of collections in './collections.json'")


if __name__ == "__main__":

    from rag_utils.db_config import EmbeddingModel, ModelsConfig
    
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    languages = ["en", "fr"]

    databases = VectorDBDataFiles.databases

    for db in databases:
        db_name = db["name"]
        model_name = selected_model.model_name + "_" + db_name.lower()

        root_path = f"outputs/{db_name}/"
        data_files_tuples = []
        
        # Create data files tuples for each type of data file
        db_ipg = db.get("ipg")
        if db_ipg:
            data_files_tuples.append((root_path + "ipgs", False))
        
        db_law = db.get("law")
        if db_law:
            for toc_type, _ in db_law[languages[0]]:
                data_files_tuples.append((root_path + toc_type, True))

        db_pages = db.get("pages")
        if db_pages:
            data_files_tuples.append((root_path + "pages", False))

        db_pdfs = db.get("pdfs")
        if db_pdfs:
            data_files_tuples.append((root_path + "pdfs", False))

        for language in languages:
            collection_name = model_name + ("_fr" if language == "fr" else "")
        
            print(f"Creating collection: {collection_name}")
            start_time = time()

            main(collection_name=collection_name,
                db_path=ChromaDBSettings.directory_path,
                data_files_tuples=data_files_tuples,
                model=selected_model,
                distance_func="ip", # passed to chromadb's get_or_create_collection method, one of ["l2", "ip", "cosine"]
                current_language=language
            )
        
            end_time = time()
            print(f"\nElapsed time for {collection_name}: ", end_time-start_time, "\n")