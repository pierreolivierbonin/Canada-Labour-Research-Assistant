from dataclasses import dataclass

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
    
    models_untested={"inf_retriever":"infly/inf-retriever-v1-1.5b", 
                     "qwen2_small":"Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                     "qwen2_large": "Alibaba-NLP/gte-Qwen2-7B-instruct",
                     "sfr_mistral": "Salesforce/SFR-Embedding-Mistral",
                     "linq_mistral":"Linq-AI-Research/Linq-Embed-Mistral"}

@dataclass
class VectorDBDataFiles:
    databases = [
    
        {
            "name": "negotech",
            "languages": ["en"],
            "pdf": {
                "en": ["https://negotech.service.canada.ca/eng/agreements/15/1538401a.pdf",
                       "https://negotech.service.canada.ca/eng/agreements/15/1538101a.pdf",
                       "https://negotech.service.canada.ca/eng/agreements/15/1537601a.pdf"]
            }
        }
    ]

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

    def assign_model_and_attributes(self):
        self.model=SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)
        self.model_chroma_callable=CustomEmbeddingFunction(model_name=self.model_name, trust_remote_code=self.trust_remote_code)
        self.max_seq_length=self.model.max_seq_length         
        self.used_seq_length=self.model.max_seq_length
        self.dimensions=self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
        
    for k, v in ModelsConfig.models.items():
        model = EmbeddingModel(model_name=ModelsConfig.models[k], trust_remote_code=True)
        model.assign_model_and_attributes()
        print(f"\nModel name: {model.model_name}")
        print(f"Model max sequence length: {model.max_seq_length}")
        print(f"Model embedding dimensions: {model.dimensions}")