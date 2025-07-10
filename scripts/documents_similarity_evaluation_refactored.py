'''
This experiment takes into account issues detailed in https://arxiv.org/pdf/2403.05440v1. 
It notably confirms the cosine distance is equal to zero (0) when the chunk is matched with itself (first distance returned). 
'''

from operator import itemgetter
import random
import sys
sys.path.append("./")

import chromadb
from chromadb.config import Settings
from scipy import stats

from config import ChromaDBSettings
from db_config import EmbeddingModel, ModelsConfig

class RAGcorpusConsistencyEvaluator:

    def __init__(self, 
                 client_path:str, 
                 embedding_model_fn:callable, 
                 collection_name:str, 
                 similarity_fn_name:str,
                 ef_construction:int,
                 threshold:int=50,
                 random_sample=False,
                 seed=123
                 ):
        
        self.client_path = client_path
        self.embedding_model_fn = embedding_model_fn
        self.collection_name = collection_name
        self.similarity_fn_name = similarity_fn_name
        self.ef_construction = ef_construction
        self.threshold = threshold
        self.random_sample = random_sample
        self.seed = seed
        
    def _initialize_client(self):
        try:
            self.client = chromadb.PersistentClient(path=self.client_path, settings=Settings(anonymized_telemetry=False))
        except Exception as e:
            print("Exception found: "+e)
        
        # try:
            # self.client = <insert your preferred vector DB here>
        # except Exception as e:
            # print(e)

    def _retrieve_all_docs(self):
        self._initialize_client()
        self.all_docs = self.collection.get()["documents"]

    def initialize_docs_collection(self):
        self._initialize_client()
        self.collection = self.client.get_or_create_collection(self.collection_name, 
                                                        embedding_function=self.embedding_model_fn,
                                                        configuration={"hnsw": {"space": self.similarity_fn_name,     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                        "ef_construction": self.ef_construction}})

    def find_similarity_score(self, save_to_disk=False):

        self._retrieve_all_docs()
        if self.random_sample:
            random.seed(1837)
            rand_ix = random.choices(range(self.collection.count()), k=100)
            self.rand_sample = itemgetter(*rand_ix)(self.all_docs)
            self.all_docs = self.rand_sample

        non_identical_matches = 0
        ip_distances_with_identical_first_match = []
        dist_description = []
        if save_to_disk:
            with open("./embedding_cosine_distance_distributions.txt", "w") as f:
                for i in self.all_docs: # switch rand_sample to all_docs when runnign full experiment

                    # search the vectorDB with one of its chunks
                    results = self.collection.query(query_texts=i, 
                                                    n_results=self.threshold, 
                                                    include=["documents", "distances"])
                    print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
                    f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                    if i!=results["documents"][0][0]: # that's where we validate that a candidate chunk and the first match are identical
                        print(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
                        non_identical_matches+=1
                    else:
                        ip_distances_with_identical_first_match.append(results["distances"])
                        dist_description.append(stats.describe(results["distances"][0]))
        else:
            for i in self.all_docs: # switch rand_sample to all_docs when runnign full experiment

                        # search the vectorDB with one of its chunks
                        results = self.collection.query(query_texts=i, 
                                                        n_results=self.threshold, 
                                                        include=["documents", "distances"])
                        print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
                        f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                        if i!=results["documents"][0][0]: # that's where we validate that a candidate chunk and the first match are identical
                            print(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
                            non_identical_matches+=1
                        else:
                            ip_distances_with_identical_first_match.append(results["distances"])
                            dist_description.append(stats.describe(results["distances"][0]))

        if save_to_disk:
            for d in dist_description:
                print(f"\n{d}")
                    
            with open(f"./documents_similarity_stats_first_{self.threshold}.csv", "w") as file:
                for d in dist_description:
                    file.writelines(str(d)+"\n")

        if self.random_sample:
            print(f"\n\nNon-identical document chunks matched for the random sample of 100 documents in collection: {non_identical_matches}")

if __name__ == "__main__":

    from db_config import EmbeddingModel, ModelsConfig

    selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)
    evaluator = RAGcorpusConsistencyEvaluator(client_path="./chroma_vectorDB_comparison",
                                             embedding_model_fn=selected_model.model_chroma_callable,
                                             collection_name="labour_baseline",
                                             similarity_fn_name="cosine",
                                             ef_construction=1000,
                                             threshold=25,
                                             random_sample=True,
                                             seed=1837
                                             )
    evaluator.initialize_docs_collection()
    evaluator.find_similarity_score(save_to_disk=True)








    # from operator import itemgetter
    # import random
    # import sys
    # sys.path.append("./")

    # import chromadb
    # from chromadb.config import Settings
    # from scipy import stats

    # from config import ChromaDBSettings
    # from db_config import EmbeddingModel, ModelsConfig

    # # First run 'create_database_with_specific_embeddings' as a standalone script (i.e. from itself so it can execute the if name block)
    # ...

    # # create client
    # client = chromadb.PersistentClient(path=ChromaDBSettings.directory_path, settings=Settings(anonymized_telemetry=False))
    # selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)

    # # Then load the collection of interest and all its chunks
    # labour_collection = client.get_or_create_collection("all-mpnet-base-v2_labour", 
    #                                                     embedding_function=selected_model.model_chroma_callable,
    #                                                     configuration={"hnsw": {"space": "cosine",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
    #                                                                     "ef_construction": 1000}})
    # collection_chunks_count = labour_collection.count()
    # all_docs = labour_collection.get()["documents"] # that's 1461 chunks

    # # Generic clone because 'embeddings' in all-mpnet-base-v2_labour are not a direct representation of 'documents' (see create_database_with_specific_embeddings.py)
    # client = chromadb.PersistentClient(path="./chroma_vectorDB_comparison", settings=Settings(anonymized_telemetry=False))
    # baseline_labour_collection = client.get_or_create_collection("labour_baseline", 
    #                                                     embedding_function=selected_model.model_chroma_callable,
    #                                                     configuration={"hnsw": {"space": "cosine",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
    #                                                                     "ef_construction": 1000}})
    
    # ## uncomment this block when creating the DB for the first time
    # # embeddings = []
    # # ids = []
    # # for ix, i in enumerate(all_docs):
    # #     # embeddings.append(selected_model.model_chroma_callable(i))
    # #     # ids.append("id"+str(ix))
    # #     try:
    # #         baseline_labour_collection.upsert(embeddings=selected_model.model_chroma_callable(i),
    # #                                         documents=i,
    # #                                         ids="id"+str(ix))
    # #         print(i[:50]+f"... successfully embedded! ({ix}/{len(all_docs)}) ---> {ix/len(all_docs)*100:.2f}% completed.")
    # #     except Exception as e:
    # #         print(e)

    # # # Let's make sure there's enough diversity in this sample: we take a random sample of 100 chunks (skip this when conducting full experiment)
    # # random.seed(1837)
    # # rand_ix = random.choices(range(collection_chunks_count), k=100)
    # # rand_sample = itemgetter(*rand_ix)(all_docs)

    # # we should expect the cosine distance between a document chunk and itself to be 0.0. Results are consistent w/ this expectation.
    # non_identical_matches = 0
    # ip_distances_with_identical_first_match = []
    # dist_description = []
    # for i in all_docs: # switch rand_sample to all_docs when runnign full experiment

    #     # search the vectorDB with one of its chunks
    #     results = baseline_labour_collection.query(query_texts=i, 
    #                                                n_results=100, 
    #                                                include=["documents", "distances"])
    #     print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
    #     if i!=results["documents"][0][0]: # that's where we validate that a candidate chunk and the first match are identical
    #         print(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
    #         non_identical_matches+=1
    #     else:
    #         ip_distances_with_identical_first_match.append(results["distances"])
    #         dist_description.append(stats.describe(results["distances"][0]))
    # print(f"\n\nNon-identical document chunks matched for the random sample of 100 documents in collection: {non_identical_matches}")

    # for d in dist_description:
    #     print(f"\n{d}")
    
    # with open("./documents_similarity_stats_first_100.csv", "w") as file:
    #     for d in dist_description:
    #         file.writelines(str(d)+"\n")

    