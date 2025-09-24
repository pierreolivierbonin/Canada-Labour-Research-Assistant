from operator import itemgetter
import random
import sys
sys.path.append("./")
from typing import Any, Callable, Dict, List, Literal
import warnings

import chromadb
from chromadb.config import Settings
from ollama import chat, ChatResponse
from pydantic import BaseModel, ValidationError, Field, model_serializer
from scipy import stats
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from db_config import EmbeddingModel, ModelsConfig


class UserInput(BaseModel):
    '''This is the actual data received.'''
    reference_chunk: str = Field(description="The reference chunk to be compared with other target chunks.")
    target_chunk: str = Field(description="The target chunk against which to compare the reference chunk. ")

# class used as a reference by the model
class ComparisonEvaluator(UserInput):
    category: Literal[
        'very high overlap', 'high overlap', 'medium overlap', 'low overlap', 'no overlap'
        ] = Field(..., description="Comparison result category.")
    reasons: str = Field(..., description="Reasoning explaining why, including:\n* query focus\n* passage focus.")
    tags: List[str] = Field(..., description="Relevant keywords related to the content of both the reference chunk and target chunk.")

    @model_serializer(when_used='json')
    def sort_model(self) -> Dict[str, Any]:
        # return dict(sorted(self.model_dump().items()))
        return {
            "reference_chunk": self.reference_chunk,
            "target_chunk": self.target_chunk,
            "category": self.category,
            "reasons": self.reasons,
            "tags": self.tags
        }
    class Config:
        """Extra configuration options"""
        anystr_strip_whitespace = True  # remove trailing whitespace

class RAGcorporaConsistencyEvaluator:

    def __init__(self, 
                 embedding_model_fn:Callable, 
                 reference_client_path:str, 
                 reference_collection_name:str, 
                 similarity_fn_name:str,
                 ef_construction:int,
                 client:Callable=chat,
                 cross_encoder:str = "cross-encoder/ms-marco-MiniLM-L6-v2",
                 target_client_path:str="",
                 target_collection_name:str="",
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

        To evaluate comparative consistency, initialize with the required arguments alongside your own target_* arguments. This will take a document chunk
        from the `reference_collection` and use it as a query with respect to the `target_collection`.

        @top_n: number of matches retrieved, in order from most to least distant (or least to most similar).
        @random_sample: if True, will not loop over the entire collection; will use a random sample specified in random_n instead
        @random_n: the size of the random sample
        @ef_construction: determines the size of the candidate list used to select neighbors during index creation. A higher value improves index quality 
                          at the cost of more memory and time, while a lower value speeds up construction with reduced accuracy. 
                          The default value is 100. [P-O: has no impact if DB already created]
        '''
        
        self.client = client
        self.embedding_model_fn = embedding_model_fn
        try:
            self.cross_encoder = CrossEncoder(cross_encoder)
        except NameError as e:
            print(e)
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
        print("\nListing all collections for reference client...\n" + str(self.reference_client.list_collections())+"\n")
        self.reference_collection = self.reference_client.get_or_create_collection(self.reference_collection_name, 
                                                        embedding_function=self.embedding_model_fn,
                                                        configuration={"hnsw": {"space": self.similarity_fn_name,     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                        "ef_construction": self.ef_construction}})
        print(f"Reference collection has length: {self.reference_collection.count()}")
        self.reference_all_docs = self.reference_collection.get()["documents"]
        self.reference_all_metadata = self.reference_collection.get()["metadatas"]
        self.reference_all_embeddings = self.reference_collection.get()["embeddings"]
        self.reference_ids = self.reference_collection.get()["ids"]

        if self.target_client_path:
            self.target_client = chromadb.PersistentClient(path=self.target_client_path, settings=Settings(anonymized_telemetry=False))
            print("\nListing all collections for target client...\n" + str(self.target_client.list_collections())+"\n")

        if self.target_client_path:
            self.target_collection = self.target_client.get_or_create_collection(self.target_collection_name, 
                                                                                 embedding_function=self.embedding_model_fn)
            self.target_all_docs = self.target_collection.get()["documents"]
            self.target_embeddings = self.target_collection.get()["embeddings"]
            self.target_ids = self.target_collection.get()["ids"]
            print(f"Target collection has length: {self.target_collection.count()}")


    def find_self_consistency_scores(self, save_to_disk=False):

        if self.random_sample:
            random.seed(1837)
            rand_ix = random.choices(range(self.reference_collection.count()), k=self.random_n)
            self.rand_sample = itemgetter(*rand_ix)(self.reference_all_docs)
        self.reference_all_docs = self.rand_sample if self.rand_sample else self.reference_all_docs

        non_identical_matches = 0
        ip_distances_with_identical_first_match = []
        dist_description = []
        self.results_compiled = []
        self.reference_cosine_dists = []
        
        for ix, i in tqdm(enumerate(self.reference_all_docs)): # switch rand_sample to all_docs when running full experiment

            # search the vectorDB with one of its own chunks
            results = self.reference_collection.query(query_texts=i, 
                                            n_results=self.top_n, 
                                            include=["metadatas", "documents", "distances"])
            if self.verbose:
                print(f"\nDocument chunk cosine distances for top-{self.top_n} matches: {[format(d, '.2f') for d in results["distances"][0]]}")
            if i!=results["documents"][0][0]: # validate that a candidate chunk and the first match are identical
                warnings.warn(f"\nInconsistent search detected - Document chunk matched with non-identical chunk\nOriginal doc: {i[:100]}\nRetrieved doc: {results["documents"][0][0][:100]}")
                non_identical_matches+=1
            else:
                ip_distances_with_identical_first_match.append(results["distances"])
                dist_description.append(stats.describe(results["distances"][0]))
            self.results_compiled.append((i, results))
            self.reference_cosine_dists.append(results["distances"][0])

        if save_to_disk:
            with open(f"./embedding_cosine_distance_distributions_random{self.random_sample}_top{self.top_n}.txt", "w") as f:
                for dist in self.reference_cosine_dists:
                    f.writelines(str([format(d, '.3f') for d in dist])+"\n")
            # f.writelines(str([format(d, '.2f') for d in results["distances"][0]])+"\n")
            with open(f"./documents_similarity_stats_top{self.top_n}.csv", "w") as file:
                for d in dist_description:
                    file.writelines(str(d)+"\n")
        if self.verbose:
            for d in dist_description:
                print(f"\n{d}")
        if self.random_sample and non_identical_matches>0:
            warnings.warn(f"\n\nNon-identical document chunks first-ranked matches for the random sample of 100 documents in collection: {non_identical_matches}")

        return self.results_compiled
    

    def find_comparative_consistency_scores(self, save_to_disk=False):
        if not (self.target_client_path and self.target_collection_name):
            raise AttributeError("Target client and collection must be initialized to conduct comparative analysis.")

        if self.random_sample:
            random.seed(1837)
            rand_ix = random.choices(range(self.reference_collection.count()), k=self.random_n)
            self.rand_sample = itemgetter(*rand_ix)(self.reference_all_docs)
        self.reference_all_docs = self.rand_sample if self.random_sample else self.reference_all_docs

        dist_description = []
        self.results_compiled = []
        self.target_cosine_dists = []

        for i,j in zip(tqdm(self.reference_all_docs), self.reference_all_metadata):

            results = self.target_collection.query(query_texts=i, 
                                            n_results=self.top_n, 
                                            include=["embeddings", "metadatas", "documents", "distances"])
            if self.verbose:
                print(f"\nDocument chunk cosine distances for top-{self.top_n} matches: {[format(d, '.2f') for d in results["distances"][0]]}")
            dist_description.append(stats.describe(results["distances"][0]))
            self.results_compiled.append({"reference": (j,i), "target": results})
            self.target_cosine_dists.append(results["distances"][0])

        if save_to_disk:
            with open(f"./embedding_cosine_distance_distributions_random{self.random_sample}_top{self.top_n}.txt", "w") as f:
                for dist in self.target_cosine_dists:
                    f.writelines(str([format(d, '.3f') for d in dist])+"\n")
            with open(f"./documents_similarity_stats_top{self.top_n}.csv", "w") as file:
                for d in dist_description:
                    file.writelines(str(d)+"\n")

        if self.verbose:
            for d in dist_description:
                print(f"\n{d}")
                    
        return self.results_compiled
    
    def call_llm(self, prompt, model='gemma3n:latest'):
        response = self.client(model=model, 
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt,
                            },
                            ])
        return response.choices[0].message.content

    def validate_with_model(self, data_model, llm_response):
        try:
            validated_data = data_model.model_validate_json(llm_response)
            print("data validation successful!")
            print(validated_data.model_dump_json(indent=2))
            return validated_data, None
        except ValidationError as e:
            print(f"error validating data: {e}")
            error_message = (
                f"This response generated a validation error: {e}."
            )
            return None, error_message


    def create_retry_prompt(self, original_prompt, original_response, error_message):
        retry_prompt = f"""
        This is a request to fix an error in the structure of an llm_response.
        Here is the original request:
        <original_prompt>
        {original_prompt}
        </original_prompt>

        Here is the original llm_response:
        <llm_response>
        {original_response}
        </llm_response>

        This response generated an error: 
        <error_message>
        {error_message}
        </error_message>

        Compare the error message and the llm_response and identify what 
        needs to be fixed or removed
        in the llm_response to resolve this error. 

        Respond ONLY with valid JSON. Do not include any explanations or 
        other text or formatting before or after the JSON string.
        """
        return retry_prompt

    def validate_llm_response(self, prompt, data_model, n_retry=5, model="gpt-4o"):
        # Initial LLM call
        response_content = RAGcorporaConsistencyEvaluator.call_llm(prompt, model=model)
        current_prompt = prompt

        # Try to validate with the model
        # attempt: 0=initial, 1=first retry, ...
        for attempt in range(n_retry + 1):

            validated_data, validation_error = self.validate_with_model(
                data_model, response_content
            )

            if validation_error:
                if attempt < n_retry:
                    print(f"retry {attempt} of {n_retry} failed, trying again...")
                else:
                    print(f"Max retries reached. Last error: {validation_error}")
                    return None, (
                        f"Max retries reached. Last error: {validation_error}"
                    )

                validation_retry_prompt = self.create_retry_prompt(
                    original_prompt=current_prompt,
                    original_response=response_content,
                    error_message=validation_error
                )
                response_content = self.call_llm(
                    validation_retry_prompt, model=model
                )
                current_prompt = validation_retry_prompt
                continue

            # If you get here, both parsing and validation succeeded
            return validated_data, None

if __name__ == "__main__":

    '''
    Related paper: https://arxiv.org/pdf/2403.05440v1. 
    The output produced  notably confirms the cosine distance is equal to zero (0) when the chunk is matched with itself (first distance returned). 
    ChromaDB's similarity metrics include: Euclidean distance, inner product, and cosine DISTANCE (misleadingly identified as cosine similarity 
    in the official doc at: https://docs.trychroma.com/docs/collections/configure) 
    
    Step 1: make sure the document chunks are embedded using 'all-mpnet-base-v2' (a.k.a. 'mpnet' in db_config) with the `get_or_create_collection()` method.
            For example: `selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)`
    Step 2: 
    '''

    import csv
    import os
    import time

    from ollama import chat, ChatResponse
    
    from sentence_transformers import CrossEncoder

    from db_config import EmbeddingModel, ModelsConfig

    selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    # Step 1 - Retrieve most similar documents based on selected metric
    evaluator = RAGcorporaConsistencyEvaluator(embedding_model_fn=selected_model.model_chroma_callable,
                                             reference_client_path="./chroma_vectorDB",
                                             reference_collection_name="multi-qa-mpnet-base-dot-v1_labour",
                                             target_client_path="./chroma_vectorDB",
                                             target_collection_name="multi-qa-mpnet-base-dot-v1_transport_act_reg",
                                             similarity_fn_name="cosine",   # should be consistent with the function used at creation time
                                             ef_construction=1000,
                                             top_n=3,
                                             random_sample=True,            # set this to False to run on entire collection (ref. or target) 
                                             random_n=5,                    # this will have no effect if random_sample=False
                                             seed=1837,
                                             verbose=True
                                             )
    
    # Step 2 - Get cosine distances
    results_comparative_consistency = evaluator.find_comparative_consistency_scores(save_to_disk=True)                                                            #   )

    # Step 3 - Save results, ordered by highest to lowest scores
    if not os.path.exists("./overlap_results/"):
        print("Warning: Folder 'overlap_results does not exist. Creating it...")
        os.mkdir("./overlap_results")

    with open("./overlap_results/results.csv", "w",  newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([
            "reference_chunk",
            "target_chunk",
            "cosine_distance",
            "cross_encoder_score",
            "category",
            "reasons",
            "keywords"
            ])
        
        for i in range(len(results_comparative_consistency)):
            start_time = time.time()

            query = results_comparative_consistency[i]["reference"][1]
            model_inputs = [[query, passage] for passage in results_comparative_consistency[i]["target"]["documents"][0]]
            scores = cross_encoder.predict(model_inputs)
            cosine_distances = results_comparative_consistency[i]["target"]["distances"][0]

            # Sort the scores in decreasing order
            results = [{"input": inp, "cosine_dist": dist, "score": score} for inp, dist, score in zip(model_inputs, cosine_distances, scores)]
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            print("\nQuery:", query[:500]+"(...)")
            print(f"\nSearch took {time.time() - start_time:.2f} seconds")
            for hit in results:
                print("\nScore: {:.2f}".format(hit["score"]), "\t", hit["input"][1][:500])


                ## Step 3.1 - Use a defined data model schema and prompt so we can get consistent output format
                input_dict = {"reference_chunk": hit["input"][0], "target_chunk": hit["input"][1]}
                validated_input = UserInput.model_validate(input_dict)

                # Step 3.2 - Ask an LLM to evaluate whether there is an overlap between passages and, if so, where it is using structured output format
                prompt = [
                    {"role":"system",
                    "content": "You are an expert at comparing legal, regulatory, and policy documents. Your task is to accurately compare two documents and extract key information based on the user-provided schema.",
                    },
                    {"role": "user",
                    "content":f"Please compare the following documents:\n\n{validated_input.model_dump_json(indent=2)}"
                    }
                    ]
                response: ChatResponse = chat(model='gemma3n:latest', 
                                            messages=prompt,
                                            format=ComparisonEvaluator.model_json_schema())
                
                final_response = ComparisonEvaluator.model_validate_json(response.message.content)
                final_response.reference_chunk = hit["input"][0]
                final_response.target_chunk = hit["input"][1]
                print(f"\n\n================LLM RESPONSE================\n{final_response.model_dump_json(indent=2)}")
                print(f"\n\nReference chunk is equal to class attribute reference chunk? {final_response.reference_chunk==hit["input"][0]}")
                print(f"\nTarget chunk is equal to class attribute target chunk? {final_response.target_chunk==hit["input"][1]}")

                writer.writerow([
                    final_response.reference_chunk,
                    final_response.target_chunk,
                    hit["cosine_dist"],
                    hit["score"],
                    final_response.category,
                    final_response.reasons,
                    final_response.tags
                    ])