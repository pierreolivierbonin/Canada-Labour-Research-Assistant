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
            "para_multiling": "paraphrase-multilingual-mpnet-base-v2",
            "multilg-e5": "intfloat/multilingual-e5-large-instruct"}
    
    models_untested={"inf_retriever":"infly/inf-retriever-v1-1.5b", 
                     "qwen2_small":"Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                     "qwen2_large": "Alibaba-NLP/gte-Qwen2-7B-instruct",
                     "sfr_mistral": "Salesforce/SFR-Embedding-Mistral",
                     "linq_mistral":"Linq-AI-Research/Linq-Embed-Mistral"}

@dataclass
class VectorDBDataFiles:
    included_databases = ["labour"] # Add equity and/or transport here, and relaunch create_or_update_database.sh, to include them.

    _databases = None
    
    # Lazily load databases from JSON files in collections folder
    @classmethod
    def databases(cls, specific_databases=None):
        if cls._databases is None:
            cls._databases = []
            collections_dir = Path("collections")
            
            if collections_dir.exists():
                # Get all .json files in collections directory
                json_files = list(collections_dir.glob("*.json"))

                # If specific_databases is provided, use it, otherwise use included_databases
                filtered_databases = specific_databases if specific_databases else cls.included_databases
                lowercase_filtered_databases = [db_name.lower() for db_name in filtered_databases]
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Only include databases that match names in filtered_databases
                            if any(db_name.lower() in data["name"].lower() for db_name in lowercase_filtered_databases):
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
        try:
            if self.model_name:
                embeddings =SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code).encode(input_)
                return embeddings
            else:
                embeddings =SentenceTransformer(self.model_card_data.base_model, trust_remote_code=self.trust_remote_code).encode(input_)
                return embeddings
        
        except Exception as e:
            print(e)




class EmbeddingModel:

    def __init__(self, model_name:str, trust_remote_code:bool=False):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code

    def assign_model_and_attributes(self):
        try:
            self.model=SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)
            if self.model_name:
                self.model_chroma_callable=CustomEmbeddingFunction(model_name=self.model_name, trust_remote_code=self.trust_remote_code)
            else:
                self.model_chroma_callable=CustomEmbeddingFunction(model_name=self.model_card_data.base_model, trust_remote_code=self.trust_remote_code)
            self.max_seq_length=self.model.max_seq_length         
            self.used_seq_length=self.model.max_seq_length
            self.dimensions=self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"\nSomething went wrong: {e}")


if __name__ == "__main__":
        
    for k, v in ModelsConfig.models.items():
        model = EmbeddingModel(model_name=ModelsConfig.models[k], trust_remote_code=True)
        model.assign_model_and_attributes()
        try:
            print(f"\nModel name: {model.model_name}")
            print(f"Model max sequence length: {model.max_seq_length}")
            print(f"Model embedding dimensions: {model.dimensions}")
        except Exception as e:
            print(e)