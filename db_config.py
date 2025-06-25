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
            "name": "labour",
            "save_html": True,
            "languages": ["en", "fr"],
            "ipg": {
                "en": "https://www.canada.ca/en/employment-social-development/programs/laws-regulations/labour/interpretations-policies.html",
                "fr": "https://www.canada.ca/fr/emploi-developpement-social/programmes/lois-reglements/travail/interpretations-politiques.html"
            },
            "law": {
                "en": [
                    ("clc", "https://laws-lois.justice.gc.ca/eng/acts/l-2/"),
                    ("clsr", "https://laws-lois.justice.gc.ca/eng/regulations/C.R.C.,_c._986/")
                ],
                "fr": [
                    ("clc", "https://laws-lois.justice.gc.ca/fra/lois/l-2/"),
                    ("clsr", "https://laws-lois.justice.gc.ca/fra/reglements/C.R.C.%2C_ch._986/")
                ]
            },
            "page": {
                "en": [
                    ("LABOUR", "https://www.canada.ca/en/employment-social-development/corporate/portfolio/labour.html"),
                    ("WORKPLACE", "https://www.canada.ca/en/services/jobs/workplace.html"),
                    ("LABOUR-REPORTS", "https://www.canada.ca/en/employment-social-development/corporate/portfolio/labour/programs/labour-standards/reports.html"),
                    ("LABOUR-STANDARDS", "https://www.canada.ca/en/services/jobs/workplace/federal-labour-standards.html"),
                    ("COMPENSATION", "https://www.canada.ca/en/services/jobs/workplace/health-safety/compensation.html"),
                    ("HEALTH-SAFETY", "https://www.canada.ca/en/services/jobs/workplace/health-safety.html")
                ],
                "fr": [
                    ("LABOUR", "https://www.canada.ca/fr/emploi-developpement-social/ministere/portefeuille/travail.html"),
                    ("WORKPLACE", "https://www.canada.ca/fr/services/emplois/milieu-travail.html"),
                    ("LABOUR-REPORTS", "https://www.canada.ca/fr/emploi-developpement-social/ministere/portefeuille/travail/programmes/normes-travail/rapports.html"),
                    ("LABOUR-STANDARDS", "https://www.canada.ca/fr/services/emplois/milieu-travail/normes-travail-federales.html"),
                    ("COMPENSATION", "https://www.canada.ca/fr/services/emplois/milieu-travail/sante-securite/indemnisation.html"),
                    ("HEALTH-SAFETY", "https://www.canada.ca/fr/services/emplois/milieu-travail/sante-securite.html")
                ]
            },
            "page_blacklist": {
                "en": [
                    "/en/news/",
                    "/en/employment-social-development/programs/laws-regulations/labour/interpretations-policies.html"
                ],
                "fr": [
                    "/fr/nouvelles.html",
                    "/fr/emploi-developpement-social/programmes/lois-reglements/travail/interpretations-politiques.html"
                ]
            }
        },
        {
            "name": "equity",
            "languages": ["en", "fr"],
            "pdf": {
                "en": [
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDWEIMSUserGuide-20220224-PDF%20(1).pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/EmployerOnboardingGuide-LEEP-2024.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/Taking%20action%20on%20your%20employment%20equity%20data.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDWorkshopHowToInterpretForm2,%20parts%20D-G-20220427.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDStep-by-step%20guide%20for%20annual%20Legislated%20Employment%20Equity%20Program%20(LEEP)%20reporting.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/EmployerOnboardingGuide-FCP-2024.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDHowToImproveWorkplaceEquity-20230308-PDF.pdf"
                ],
                "fr": [
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDWEIMSUserGuideFR-20220224-PDF%20(1).pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/GuideDIntegrationPourLesEmployeurs-PLEME-2024.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/Agir%20a%20partir%20de%20vos%20donnees%20sur%20l%20equite%20en%20emploi.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDAtelierCommentInterpreterVotreFormulaire%202,%20parties%20D-G-20220428.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-Guide%20de%20preparation%20etape%20par%20etape%20du%20rapport%20annuel%20du%20Programme%20legifere%20dequite%20en%20matiere%20demploi%20(PLEME).pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/GuideDIntegrationPourLesEmployeurs-PCF-2024.pdf",
                    "https://equity.esdc.gc.ca/sgiemt-weims/maint/file/download/FP-GC-WEDCommentAmeliorerLequiteSurLeLieuDuTravailFR-20230308-PDF.pdf"
                ]
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