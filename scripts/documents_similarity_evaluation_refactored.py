from operator import itemgetter
import random
import sys
sys.path.append("./")
import warnings

import chromadb
from chromadb.config import Settings
from scipy import stats

from db_config import EmbeddingModel, ModelsConfig

class RAGcorporaConsistencyEvaluator:

    def __init__(self, 
                 embedding_model_fn:callable, 
                 reference_client_path:str, 
                 reference_collection_name:str, 
                 similarity_fn_name:str,
                 ef_construction:int,
                 target_client_path:str=None,
                 target_collection_name:str=None,
                 top_n:int=50,
                 random_sample=False,
                 random_n=100,
                 seed=123,
                 verbose=True
                 ):
        
        '''
        This class evaluates the consistency of a corpus either 1) with itself, or 2) with a different corpus.
        Useful for applications leveraging Retrieval-Augmented Generation.

        To evaluate self-consistency, initialize with the default target_* parameters. This will take a document chunk 
        from the `reference_collection` and use it as a query with respect to its own corpus.

        To evaluate comparative consistency, initialize with the required arguments alongside target_* arguments. This will take a document chunk
        from the `reference_collection` and use it as a query with respect to the `target_collection`.

        @top_n: number of matches retrieved, in order from most to least to most distant (or least to most similar).
        @random_sample: if True, will not loop over the entire collection; will use a random sample specified in random_n instead
        @random_n: the size of the random sample
        '''
        
        self.embedding_model_fn = embedding_model_fn
        self.reference_client_path = reference_client_path
        self.reference_collection_name = reference_collection_name
        self.target_client_path = target_client_path
        self.target_collection_name = target_collection_name
        self.similarity_fn_name = similarity_fn_name
        self.ef_construction = ef_construction
        self.top_n = top_n
        self.random_sample = random_sample
        self.random_n = random_n
        self.seed = seed
        self.verbose = verbose

        self.reference_client = chromadb.PersistentClient(path=self.reference_client_path, settings=Settings(anonymized_telemetry=False))
        self.reference_collection = self.reference_client.get_or_create_collection(self.reference_collection_name, 
                                                        embedding_function=self.embedding_model_fn,
                                                        configuration={"hnsw": {"space": self.similarity_fn_name,     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                        "ef_construction": self.ef_construction}})
        
        if self.target_client_path:
            self.target_client = chromadb.PersistentClient(path=self.target_client_path, settings=Settings(anonymized_telemetry=False))
            print("\nListing all collections...\n" + str(self.target_client.list_collections())+"\n")

        if self.target_client_path:
            self.target_collection = self.target_client.get_collection(self.target_collection_name, 
                                                        embedding_function=self.embedding_model_fn)

    def _retrieve_all_docs(self):
        self.all_docs = self.reference_collection.get()["documents"] # that's 1461 chunks

    def _retrieve_all_embeddings(self):
        self.all_embeddings = self.reference_collection.get()["embeddings"]

    def find_self_consistency_scores(self, save_to_disk=False):
        self._retrieve_all_docs()
        if self.random_sample:
            random.seed(1837)
            rand_ix = random.choices(range(self.reference_collection.count()), k=self.random_n)
            self.rand_sample = itemgetter(*rand_ix)(self.all_docs)
            self.all_docs = self.rand_sample

        non_identical_matches = 0
        ip_distances_with_identical_first_match = []
        dist_description = []
        if save_to_disk:
            with open(f"./embedding_cosine_distance_distributions_random{self.random_sample}_top{self.top_n}.txt", "w") as f:
                self.results_compiled = []
                for ix, i in enumerate(self.all_docs): # switch rand_sample to all_docs when running full experiment

                    # search the vectorDB with one of its own chunks
                    results = self.reference_collection.query(query_texts=i, 
                                                    n_results=self.top_n, 
                                                    include=["documents", "distances"])
                    if self.verbose:
                        print(f"\nDocument chunk cosine distances for top-{self.top_n} matches: {[format(d, '.2f') for d in results["distances"][0]]}")
                    f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                    if i!=results["documents"][0][0]: # validate that a candidate chunk and the first match are identical
                        print(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
                        non_identical_matches+=1
                    else:
                        ip_distances_with_identical_first_match.append(results["distances"])
                        dist_description.append(stats.describe(results["distances"][0]))
                    self.results_compiled.append((i, results))
            
        else:
            self.results_compiled = []
            for ix, i in enumerate(self.all_docs): # switch rand_sample to all_docs when runnign full experiment

                # search the vectorDB with one of its own chunks
                results = self.reference_collection.query(query_texts=i, 
                                                n_results=self.top_n, 
                                                include=["documents", "distances"])
                if self.verbose:
                    print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
                f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                if i!=results["documents"][0][0]: # that's where we validate that a candidate chunk and the first match are identical
                    warnings.warn(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
                    non_identical_matches+=1
                else:
                    ip_distances_with_identical_first_match.append(results["distances"])
                    dist_description.append(stats.describe(results["distances"][0]))
                self.results_compiled.append((i, results))

        if self.verbose:
            for d in dist_description:
                print(f"\n{d}")
                    
            with open(f"./documents_similarity_stats_top{self.top_n}.csv", "w") as file:
                for d in dist_description:
                    file.writelines(str(d)+"\n")

        if self.random_sample and non_identical_matches>0:
            warnings.warn(f"\n\nNon-identical document chunks matched for the random sample of 100 documents in collection: {non_identical_matches}")

        return self.results_compiled
    

    def find_comparative_consistency_scores(self, save_to_disk=False):
        if not (self.target_client_path and self.target_collection_name):
            raise AttributeError("Target client and collection must be initialized to conduct comparative analysis.")
        self._retrieve_all_docs()
        if self.random_sample:
            random.seed(1837)
            rand_ix = random.choices(range(self.reference_collection.count()), k=self.random_n)
            self.rand_sample = itemgetter(*rand_ix)(self.all_docs)
            self.all_docs = self.rand_sample

        dist_description = []
        if save_to_disk:
            with open(f"./embedding_cosine_distance_distributions_random{self.random_sample}_top{self.top_n}.txt", "w") as f:
                self.results_compiled = []
                for ix, i in enumerate(self.all_docs): # switch rand_sample to all_docs when running full experiment

                    # search the vectorDB with one of its own chunks
                    results = self.target_collection.query(query_texts=i, 
                                                    n_results=self.top_n, 
                                                    include=["documents", "distances"])
                    if self.verbose:
                        print(f"\nDocument chunk cosine distances for top-{self.top_n} matches: {[format(d, '.2f') for d in results["distances"][0]]}")
                    f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                    dist_description.append(stats.describe(results["distances"][0]))
                    self.results_compiled.append((i, results))
            
        else:
            self.results_compiled = []
            for ix, i in enumerate(self.all_docs): # switch rand_sample to all_docs when runnign full experiment

                # search the vectorDB with one of its own chunks
                results = self.reference_collection.query(query_texts=i, 
                                                n_results=self.top_n, 
                                                include=["documents", "distances"])
                if self.verbose:
                    print(f"\nDocument chunk cosine distances: {[format(d, '.2f') for d in results["distances"][0]]}")
                f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")

                dist_description.append(stats.describe(results["distances"][0]))
                self.results_compiled.append((i, results))

        if self.verbose:
            for d in dist_description:
                print(f"\n{d}")
                    
            with open(f"./documents_similarity_stats_top{self.top_n}.csv", "w") as file:
                for d in dist_description:
                    file.writelines(str(d)+"\n")

        return self.results_compiled


if __name__ == "__main__":

    '''
    Related paper: https://arxiv.org/pdf/2403.05440v1. 
    The output produced  notably confirms the cosine distance is equal to zero (0) when the chunk is matched with itself (first distance returned). 
    '''

    from db_config import EmbeddingModel, ModelsConfig

    selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)

    ## run on a random sample of 25 document chunks
    # evaluator = RAGcorporaConsistencyEvaluator(embedding_model_fn=selected_model.model_chroma_callable,
    #                                          reference_client_path="./chroma_vectorDB_comparison",
    #                                          reference_collection_name="labour_baseline",
    #                                          target_client_path="./chroma_vectorDB",
    #                                          target_collection_name="all-mpnet-base-v2_labour",
    #                                          similarity_fn_name="cosine",
    #                                          ef_construction=1000,
    #                                          top_n=10,
    #                                          random_sample=True,
    #                                          random_n=25,
    #                                          seed=1837
    #                                          )
    
    # results_self_consistency = evaluator.find_self_consistency_scores(save_to_disk=True)

    # # manual validation for self-consistency: 
    # # How distant are the queried document chunks from the retrieved documents chunks of the same collection?
    # for ix, result in enumerate(results_self_consistency):
    #     print(f"\nDocument queried... \n\n{result[0]}")
    #     print(f"\nPreviewing top-{evaluator.top_n} matches...")
        
    #     for jx, j in enumerate(range(len(result[1]["documents"][0]))):
    #         print(f"\nViewing matched Document-chunk rank #{jx+1}... \n...for Document-chunk query #{ix+1}...")
    #         print(f"\n{result[1]["documents"][0][j]}")

    # # manual validation for comparative consistency: 
    # # How distant are the queried chunks of the reference collection from the retrieved chunks of the target collection?
    # results_comparative_consistency = evaluator.find_comparative_consistency_scores(save_to_disk=True)

    # for ix, result in enumerate(results_comparative_consistency):
    #     print(f"\nDocument queried... \n\n{result[0]}")
    #     print(f"\nPreviewing top-{evaluator.top_n} matches...")
        
    #     for jx, j in enumerate(range(len(result[1]["documents"][0]))):
    #         print(f"\nViewing matched Document-chunk rank #{jx+1}... \n...for Document-chunk query #{ix+1}...")
    #         print(f"\n{result[1]["documents"][0][j]}")

    # run on the entire collection
    evaluator = RAGcorporaConsistencyEvaluator(embedding_model_fn=selected_model.model_chroma_callable,
                                             reference_client_path="./chroma_vectorDB_comparison",
                                             reference_collection_name="labour_baseline",
                                             target_client_path="./chroma_vectorDB",
                                             target_collection_name="all-mpnet-base-v2_labour",
                                             similarity_fn_name="cosine",
                                             ef_construction=1000,
                                             top_n=73,                        # total chunks = 1461, so 1461*0.05==73 for top 5% matches
                                             random_sample=False,
                                             seed=1837
                                             )
    
    evaluator.find_self_consistency_scores(save_to_disk=True)