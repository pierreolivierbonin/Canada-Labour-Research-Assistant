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
    from scipy import stats

    labour_collection = _load_vector_database(language="en", db_name="Labour")
    collection_chunks_count = labour_collection.count()

    all_docs = labour_collection.get()["documents"]

    # Let's make sure there's enough diversity in this sample
    rnd_ix = random.choices(range(collection_chunks_count+1), k=100)
    rnd_sample = itemgetter(*rnd_ix)(all_docs)

    non_identical_matches = 0
    ip_distances_with_identical_first_match = []
    dist_description = []
    for i in rnd_sample:
        docs, metadata, ids, distances = fetch_documents_from_database_simplified(i, language="en", db_name="Labour", n_results=50)
        print(f"Document chunk cosine distances: {[format(d, '.2f') for d in distances]}")
        if i[:100]!=docs[0][:100]: # that's where we determine identity
            print(f"Original doc: {i[:100]}\nRetrieved doc: {docs[0][:100]}")
            non_identical_matches+=1
        else:
            ip_distances_with_identical_first_match.append(distances)
            dist_description.append(stats.describe(distances))
    print(f"\n\nNon-identical document chunks matched for the random sample of 100 documents in collection: {non_identical_matches}")

    # for d in dist_description:
    #     print(f"\n{d}")
    
    # with open("./documents_similarity_stats.csv", "w") as file:
    #     for d in dist_description:
    #         file.writelines(str(d)+"\n")