from dataclasses import dataclass
import json
from pathlib import Path

from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

'''
NOTE: Found out that dataclasses are getting initialized as soon as we import this module, or 
    if we run this script. Workaround: outsource the attributes that we don't want 
    to initialize right away.
'''

'''
The EmbeddingModelsTesting.requires_validation models are the top 5 models scoring 
the highest on the MTEB, using the following filters:

-- Prebuilt benchmark: MTEB(Multilingual, v1)
-- Languages: all
-- Task types: InstructionRetrieval, Reranking, Retrieval, STS 
-- Domains: academic, encyclopaedic, government, legal, subtitles, web, written
-- Added and removed tasks: default values.
'''

@dataclass
class ModelsConfig:
    models={"multi_qa":"multi-qa-mpnet-base-dot-v1",
            "mpnet":"all-mpnet-base-v2", 
            "biling_lg":"Lajavaness/bilingual-embedding-large"}
    
    models_similarity_fn={"multi-qa-mpnet-base-dot-v1":"ip",
                          "all-mpnet-base-v2":"cosine",
                          "Lajavaness/bilingual-embedding-large":"N/A"}
    
    models_untested={"inf_retriever":"infly/inf-retriever-v1-1.5b", 
                     "qwen2_small":"Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                     "qwen2_large": "Alibaba-NLP/gte-Qwen2-7B-instruct",
                     "sfr_mistral": "Salesforce/SFR-Embedding-Mistral",
                     "linq_mistral":"Linq-AI-Research/Linq-Embed-Mistral"}

@dataclass
class VectorDBDataFiles:
    _databases = None
    
    # Lazily load databases from JSON files in collections folder
    @classmethod
    def databases(cls):
        if cls._databases is None:
            cls._databases = []
            collections_dir = Path("collections")
            
            if collections_dir.exists():
                # Get all .json files in collections directory
                json_files = list(collections_dir.glob("*.json"))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            cls._databases.append(data)
                    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
                        print(f"Warning: Could not load {json_file}: {e}")
                        continue
        
        return cls._databases

@dataclass
class FilteredMTEB:
    url="http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28Multilingual%2C+v1%29"
    additional_filters=["ordered by retrieval score", "availability=open only"]

class CustomEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name, trust_remote_code=False):
        super().__init__()
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
    
    def __call__(self, input_):
        embeddings =SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code).encode(input_)
        return embeddings


class EmbeddingModel:

    def __init__(self, model_name:str, trust_remote_code:bool=False):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.model=SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)
        self.model_chroma_callable=CustomEmbeddingFunction(model_name=self.model_name, trust_remote_code=self.trust_remote_code)

        try:
            self.max_seq_length=self.model.max_seq_length
        except Exception as e:
            print(e)
        try:
            self.used_seq_length=self.model.max_seq_length
        except Exception as e:
            print(e)
        try:
            self.dimensions=self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(e)
        if self.model.similarity_fn_name is None:
            self.model.similarity_fn_name = ModelsConfig.models_similarity_fn[model_name]
            print(f"Model similarity function name: {self.model.similarity_fn_name}")


if __name__ == "__main__":
        
    for k, v in ModelsConfig.models.items():
        try:
            model_instance = EmbeddingModel(model_name=v, trust_remote_code=True)
            print(f"\nModel name: {model_instance.model_name}")
            print(f"Model max sequence length: {model_instance.max_seq_length}")
            print(f"Model embedding dimensions: {model_instance.dimensions}")
            if model_instance.model.similarity_fn_name is not None:
                print(f"Model similarity function name: {model_instance.model.similarity_fn_name}")
        except Exception as e:
            print("\n"+ str(e))
        if model_instance.model.similarity_fn_name is None:
            model_instance.similarity_fn_name = ModelsConfig.models_similarity_fn[k]
            print(f"Model similarity function name: {model_instance.similarity_fn_name}")