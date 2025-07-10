'''
This experiment takes into account issues detailed in https://arxiv.org/pdf/2403.05440v1
'''


if __name__ == "__main__":

    from operator import itemgetter
    import random
    import sys
    sys.path.append("./")

    import chromadb
    from chromadb.config import Settings
    from scipy import stats

    from config import ChromaDBSettings
    from db_config import EmbeddingModel, ModelsConfig

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

    # Generic clone because 'embeddings' in all-mpnet-base-v2_labour are not a direct representation of 'documents' (see create_database_with_specific_embeddings.py)
    client = chromadb.PersistentClient(path="./chroma_vectorDB_comparison", settings=Settings(anonymized_telemetry=False))
    baseline_labour_collection = client.get_or_create_collection("labour_baseline", 
                                                        embedding_function=selected_model.model_chroma_callable,
                                                        configuration={"hnsw": {"space": "cosine",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                        "ef_construction": 1000}})
    
    ## uncomment this block when creating the DB for the first time
    # embeddings = []
    # ids = []
    # for ix, i in enumerate(all_docs):
    #     # embeddings.append(selected_model.model_chroma_callable(i))
    #     # ids.append("id"+str(ix))
    #     try:
    #         baseline_labour_collection.upsert(embeddings=selected_model.model_chroma_callable(i),
    #                                         documents=i,
    #                                         ids="id"+str(ix))
    #         print(i[:50]+f"... successfully embedded! ({ix}/{len(all_docs)}) ---> {ix/len(all_docs)*100:.2f}% completed.")
    #     except Exception as e:
    #         print(e)

    # # Let's make sure there's enough diversity in this sample: we take a random sample of 100 chunks (skip this when conducting full experiment)
    # random.seed(1837)
    # rand_ix = random.choices(range(collection_chunks_count), k=100)
    # rand_sample = itemgetter(*rand_ix)(all_docs)

    # we should expect the cosine distance between a document chunk and itself to be 0.0. Results are consistent w/ this expectation.
    non_identical_matches = 0
    ip_distances_with_identical_first_match = []
    dist_description = []
    for i in all_docs: # switch rand_sample to all_docs when runnign full experiment

        # search the vectorDB with one of its chunks
        results = baseline_labour_collection.query(query_texts=i, 
                                                   n_results=100, 
                                                   include=["documents", "distances"])
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
    
    with open("./documents_similarity_stats_first_100.csv", "w") as file:
        for d in dist_description:
            file.writelines(str(d)+"\n")

    