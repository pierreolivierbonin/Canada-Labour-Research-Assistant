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
                 top_n:int=50,
                 random_sample=False,
                 random_n=100,
                 seed=123
                 ):
        
        self.client_path = client_path
        self.embedding_model_fn = embedding_model_fn
        self.collection_name = collection_name
        self.similarity_fn_name = similarity_fn_name
        self.ef_construction = ef_construction
        self.top_n = top_n
        self.random_sample = random_sample
        self.random_n = random_n
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
            rand_ix = random.choices(range(self.collection.count()), k=self.random_n)
            self.rand_sample = itemgetter(*rand_ix)(self.all_docs)
            self.all_docs = self.rand_sample

        non_identical_matches = 0
        ip_distances_with_identical_first_match = []
        dist_description = []
        if save_to_disk:
            with open(f"./embedding_cosine_distance_distributions_random{self.random_sample}_top{self.top_n}.txt", "w") as f:
                for i in self.all_docs: # switch rand_sample to all_docs when runnign full experiment

                    # search the vectorDB with one of its chunks
                    results = self.collection.query(query_texts=i, 
                                                    n_results=self.top_n, 
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
                                                        n_results=self.top_n, 
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
                    
            with open(f"./documents_similarity_stats_top{self.top_n}.csv", "w") as file:
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
                                             top_n=10,
                                             random_sample=True,
                                             random_n=25,
                                             seed=1837
                                             )
    evaluator.initialize_docs_collection()
    evaluator.find_similarity_score(save_to_disk=True)