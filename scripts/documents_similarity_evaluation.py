import sys
sys.path.append("./")

from tools import _load_vector_database


def fetch_documents_from_database_simplified(database_question, 
                                  language,
                                  db_name,
                                  n_results=5):

    results = _load_vector_database(language, db_name).query(query_texts=database_question,
                                            n_results=n_results,
                                            include=["metadatas", "distances", "documents"])

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    ids = [str(i) for i in results["ids"][0]]
    distances = results["distances"][0]

    return documents, metadata, ids, distances


if __name__ == "__main__":

    from operator import itemgetter
    import random

    import chromadb
    from chromadb.config import Settings
    from scipy import stats

    from config import ChromaDBSettings
    from db_config import EmbeddingModel, ModelsConfig
    from tools import _load_vector_database

    # First run 'create_database_with_specific_embeddings' as a standalone script (i.e. from itself so it can execute the if name block)
    ...

    # create client
    client = chromadb.PersistentClient(path=ChromaDBSettings.directory_path, settings=Settings(anonymized_telemetry=False))
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)

    # Then load the collection of interest and all its chunks
    labour_collection = client.get_or_create_collection("all-mpnet-base-v2_labour", 
                                                        embedding_function=selected_model.model_chroma_callable,
                                                        configuration={"hnsw": {"space": "cosine",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                        "ef_construction": 1000}})
    collection_chunks_count = labour_collection.count()

    all_docs = labour_collection.get()["documents"] # that's 1461 chunks

    # Let's make sure there's enough diversity in this sample: we take a random sample of 100 chunks
    random.seed(1837)
    rand_ix = random.choices(range(collection_chunks_count), k=100)
    rand_sample = itemgetter(*rand_ix)(all_docs)

    # we should expect the cosine distance between a document chunk and itself to be 0.0
    non_identical_matches = 0
    ip_distances_with_identical_first_match = []
    dist_description = []
    for i in rand_sample:
        results = labour_collection.query(query_texts=i, n_results=50)
        print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
        if i!=results["documents"][0][0]: # that's where we validate that a candidate chunk and the first match are identical
            print(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
            non_identical_matches+=1
        else:
            ip_distances_with_identical_first_match.append(results["distances"])
            dist_description.append(stats.describe(results["distances"][0]))
    print(f"\n\nNon-identical document chunks matched for the random sample of 100 documents in collection: {non_identical_matches}")

    for d in dist_description:
        print(f"\n{d}")
    
    with open("./documents_similarity_stats.csv", "w") as file:
        for d in dist_description:
            file.writelines(str(d)+"\n")