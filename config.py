from dataclasses import dataclass, field
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class CustomEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name, trust_remote_code=False):
        super().__init__()
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
    
    def __call__(self, input_):
        embeddings =SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code).encode(input_)
        return embeddings

@dataclass
class ChromaDBSettings:
    directory_path="./chroma_vectorDB"
    embedding_model="multi-qa-mpnet-base-dot-v1"
    anonymized_telemetry=False
    collections = {"bilingual_embed_large": "Mar082025_BilingualEmbedLarge",
                    "multi_qa_mpnet":"multi-qa-mpnet-base-dot-v1",
                   "snowflake_arctic":"Labour_Program_Feb192025_snowflake_arctic_embed_m_v2"}
    nb_docs_to_prioritize_multiplier = 3
    nb_docs_returned = 5
    trust_remote_code=True

@dataclass
class BaseChatbotInterfaceConfig:
    default_mode: str = ""
    title: str = ""
    default_model_local: str = ""
    models_shortlist_local: list = field(default_factory=list)
    default_model_remote: str = ""
    models_shortlist_remote: list = field(default_factory=list)
    language: str = ""
    nb_previous_questions: int = 0
    db_name: str = ""

@dataclass
class ChatbotInterfaceConfig(BaseChatbotInterfaceConfig):
    default_mode = "local"
    title = "Canada Labour Research Assistant"
    default_model_local = "llama3.2:latest"
    models_shortlist_local = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b',
                        'granite3-dense:8b',
                        'llama3.2:latest', "llama3.2-3B-instruct-q4-k-l:latest", "llama3.2-3B-instruct-q4-k-m:latest",
                        'mistral-small:24b-instruct-2501-q4_K_M', 'mistral-small3.1', 'mistral-nemo']
    default_model_remote = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    models_shortlist_remote = [default_model_remote]
    language = "en"
    nb_previous_questions = 1
    db_name = "labour"

@dataclass
class vLLMChatbotInterfaceConfig(BaseChatbotInterfaceConfig):
    default_mode = "local"
    title = "Canada Labour Research Assistant"
    default_model_local = "meta-llama/Llama-3.2-3B-Instruct"
    models_shortlist_local = ["meta-llama/Llama-3.2-3B-Instruct"] #, "Llama-3.2-3B-Instruct-W4A16-G128"]
    default_model_remote = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    models_shortlist_remote = ["meta-llama/Llama-4-Scout-17B-16E-Instruct"]
    language = "en"
    nb_previous_questions = 1
    db_name = "labour"

@dataclass
class Evaluation:
    summac={"models":["vitc"],
            "bins":'percentile',
            'granularity':'sentence',
            'nli_labels':'e',
            'device':'cuda',
            'start_file':'.models/summac_conv_vitc_sent_perc_e.bin',
            'agg':'mean'}

@dataclass
class QuotationsConfig:
    threshold_rouge_score = 0.5 # Determine how close to the LCS in the source documents a quote has to be in order to be replaced by it (that is, in order to create a colored citation).
    min_non_header_words_in_quote = 9
    direct_quotations_mode = True

@dataclass
class PromptTemplateType:
    minimalist: str = """You are a helpful assistant. 
You recognize when you do not know the answer and ask for clarifications when needed."""
    structured_with_context_no_direct_quotations: str = """Answer the question based on the previous documents.
Structure your responses with section headers and subtitles."""
    # Avoid adding unecessary spaces in the prompt
    structured_with_context_local: str = """Answer the question based on the previous documents.
Structure your responses with section headers and subtitles.
Use quotation marks to directly quote relevant passages from the text, giving an in depth analysis of how the quote relates to the question.
Include at least 1 quotation in your answer, ideally more.
Do not refer to examples from the source documents, unless you quote them first.
Those passages should be quoted word for word, avoiding the use of ellipsis to shorten the quote.
Avoid quoting passages inline with your text, instead quote them on a new line.
Do not quote the same passage twice.
Do not list the quotes you used at the end of your answer.
"""
    structured_with_context_remote: str = """Answer the question based on the previous source documents.
Structure your responses with section headers and subtitles.
Use quotation marks to directly quote relevant passages from the text, giving an in depth analysis of how the quote relates to the question.
Those passages should be quoted word for word, avoiding the use of ellipsis to shorten the quote.
Avoid quoting passages inline with your text, instead quote them on a new line.
You should mention the source of the quote before quoting it.
Do not quote the same passage twice.
Do not list the quotes you used at the end of your answer.
"""
    answer_in_french: str = "Répondez à la question en français."
    question_intro_en: str = "HERE IS THE QUESTION"
    question_intro_fr: str = "VOICI LA QUESTION"
    previous_question_en: str = "previous question"
    previous_question_fr: str = "question précédente"
    previous_answer_en: str = "previous answer"
    previous_answer_fr: str = "réponse précédente"
    message_template_token_count = 24 # Every call to the LLM will have this many tokens added to the prompt due to the message template (ex: {"role": "user", "content": "..."}) (doesn't include the 1 extra for the start token)

@dataclass
class ConsoleConfig:
    verbose = True

@dataclass
class RAGConfig:
    '''
    Maximum context length for all models.
    '''

    model_max_context_length = {
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": 327680,
        "meta-llama/Llama-3.2-3B-Instruct": 131072,
        "google/gemma-2-9b": 131072,
        "gemma3:1b": 32768,
        "gemma3:4b": 131072,
        "gemma3:12b": 131072,
        "granite3-dense:8b": 4096,
        "llama3.2:latest": 131072,
        "llama3.2-3B-instruct-q4-k-l:latest": 131072,
        "llama3.2-3B-instruct-q4-k-m:latest": 131072,
        "Llama-3.2-3B-Instruct-W4A16-G128": 131072,
        "mistral-small:24b-instruct-2501-q4_K_M": 32768,
        "mistral-small3.1": 131072,
        "mistral-nemo": 128000
    }
    # Maximum allowed context size in tokens for all models (unless the model has a lower limit). 
    # Will never go above this limit, even if the model can handle more.
    max_allowed_context_size_local = 32768 
    max_allowed_context_size_remote = None # None means no limit

@dataclass
class OllamaRAGConfig:
    '''
    See Ollama's accepted parameters https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    '''

    HyperparametersAccuracyConfig = {
        "mirostat_tau":0,
        "seed":1837,
        "num_ctx": 4096, # context window used (4096 covers the default value of 5 chunks of 500 tokens each retrieved + buffer); impacts latency on lower-grade GPUs
        "temperature": 0.0,
        "top_k":1,
        "top_p":0.1 # Top P is not used unless you set the Top P parameter value to something other than the default value of 1.
    }

@dataclass
class vLLMRAGConfig:
    '''
    See vLLM's accepted parameters list: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#id7   
    '''

    HyperparametersAccuracyConfig = {
        "n":1,
        "seed":1837,
        "temperature":0.0,
        "top_p":0.1,
        # these params have to be passed in "extra_body" as per https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#id7    
        "extra_body":{
        #  "repetition_penalty":0.5,
        #  "frequency_penalty":0.5,
            "echo":False,
            "top_k":1,
            "max_tokens":500 # Maximum number of tokens to generate per output sequence (https://docs.vllm.ai/en/v0.6.0/dev/sampling_params.html)
        }
    }

    # the following arguments are passed when the app is launched only; relaunch the app once you change the values
    EngineArgs = {
        "model_name":"meta-llama/Llama-3.2-3B-Instruct",
        "ctx_window":8000,
        "gpu_memory_utilization":0.95,
        "max_num_batched_tokens":8000,
        "max_num_seqs":2
    }