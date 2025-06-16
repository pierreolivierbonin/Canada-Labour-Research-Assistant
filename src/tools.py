import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory to sys.path
import time

import chromadb
from chromadb.config import Settings
import streamlit as st
import torch

from config import ChatbotInterfaceConfig, ChromaDBSettings, CustomEmbeddingFunction, PromptTemplateType, RAGConfig, OllamaRAGConfig, QuotationsConfig, ConsoleConfig
from local import get_ollama_answer_local, get_ollama_answer_local_stream
from local_vllm import get_vllm_answer, get_vllm_answer_stream
from remote import get_llm_answer_remote, get_llm_answer_remote_stream
from paragraph_generator import markdown_post_processing, post_processing, get_paragraph_generator
from context import manage_max_context_length, extract_reference_section_numbers, format_context_for_prompt, format_for_documents_tab_ui, format_for_metadata_tab_ui, reprioritize_docs

torch.classes.__path__ = [] # this speeds up launch time by fixing an unresolved issue with Streamlit, as per https://discuss.streamlit.io/t/message-error-about-torch/90886/5

@st.cache_resource(show_spinner="Loading Database...")
def _load_vector_database(language, 
                          db_name,
                          database_storage_path=ChromaDBSettings.directory_path,
                          collection_name=ChromaDBSettings.collections["multi_qa_mpnet"],     # Labour_Standards_IPGs with all-MiniLM-L12 works well
                          embedding_model=ChromaDBSettings.embedding_model,
                          trust_remote_code=ChromaDBSettings.trust_remote_code):                  # all-MiniLM-L12-v2 or multi-qa-MiniLM-L6-dot-v1 for 384-dim; emb dims must match
    
    client = chromadb.PersistentClient(path=database_storage_path, 
                                       settings=Settings(anonymized_telemetry=False))
    
    sentence_transformer_ef = CustomEmbeddingFunction(model_name=embedding_model, trust_remote_code=trust_remote_code)

    collection_name_in_language = collection_name + "_" + db_name.lower() + ("_" + language if language != "en" else "")

    collection = client.get_collection(collection_name_in_language, 
                                       embedding_function=sentence_transformer_ef)
    
    return collection

def fetch_documents_from_database(database_question, 
                                  language,
                                  db_name,
                                  question_section_numbers,
                                  n_results):
    nb_docs_to_prioritize_multiplier = ChromaDBSettings.nb_docs_to_prioritize_multiplier if len(question_section_numbers) > 0 else 1
    nb_docs_to_fetch = n_results * nb_docs_to_prioritize_multiplier

    results = _load_vector_database(language, db_name).query(query_texts=database_question,
                                            n_results=nb_docs_to_fetch,
                                            include=["metadatas", "distances", "documents"])

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    ids = [str(i) for i in results["ids"][0]]
    distances = results["distances"][0]

    # If there are question section numbers, also fetch documents that match the question section numbers (gives it low priority, just in case no documents related to them are found initially)
    if len(question_section_numbers) > 0:
        results_sections = _load_vector_database(language, db_name).query(query_texts=database_question,
                                                n_results=n_results,
                                                include=["metadatas", "distances", "documents"],
                                                where={"$or": [
                                                    {"section_number": {"$in": question_section_numbers}},
                                                    {"main_section_number": {"$in": question_section_numbers}}
                                                ]})
        
        for index, id in enumerate(results_sections["ids"][0]):
            if id not in ids:
                documents.append(results_sections["documents"][0][index])
                metadata.append(results_sections["metadatas"][0][index])
                ids.append(id)
                distances.append(results_sections["distances"][0][index])

    return documents, metadata, ids, distances

# Get the prompt template based on whether the model is remote or not
def get_prompt_template(custom_system_prompt, is_remote, quotations_mode, language):
    if not quotations_mode:
        prompt_template = PromptTemplateType.structured_with_context_no_direct_quotations
    elif is_remote:
        prompt_template = PromptTemplateType.structured_with_context_remote
    else:
        prompt_template = PromptTemplateType.structured_with_context_local

    if language == "fr":
        prompt_template += PromptTemplateType.answer_in_french

    if custom_system_prompt:
        prompt_template += "\n\n" + custom_system_prompt

    return prompt_template

# @st.cache_data(show_spinner=False, ttl=600)
def retrieve_database(database_question,
                      language,
                      db_name,
                      is_remote,
                      chat_model,
                      n_results,
                      hyperparams,
                      quotations_mode,
                      custom_system_prompt,
                      previous_messages,
                      previous_question_chunks,
                      nb_previous_questions=1):

    if chat_model is None:
        chat_model = ChatbotInterfaceConfig.default_model_local if not is_remote else ChatbotInterfaceConfig.default_model_remote

    # Clone the hyperparams and find the max context length based on the model
    hyperparams = hyperparams.copy()
    model_max_context_length = RAGConfig.model_max_context_length.get(chat_model)
    max_allowed_context_size = RAGConfig.max_allowed_context_size_local if not is_remote else RAGConfig.max_allowed_context_size_remote

    if model_max_context_length is None:
        print(f"WARNING: Model {chat_model} not found in RAGConfig.model_max_context_length")
        max_context_length = max_allowed_context_size
    elif max_allowed_context_size is None:
        max_context_length = model_max_context_length
    else:
        max_context_length = min(model_max_context_length, max_allowed_context_size)

    prompt_template_type = get_prompt_template(custom_system_prompt, is_remote, quotations_mode, language)

    question_section_numbers = extract_reference_section_numbers(database_question)
    documents, metadata, ids, distances = fetch_documents_from_database(
        database_question, language, db_name,question_section_numbers, n_results
    )
    
    formatted_documents= format_context_for_prompt(ids, metadata, documents)
    formatted_documents, metadata, ids, distances = reprioritize_docs(
        question_section_numbers, formatted_documents, metadata, ids, distances
    )

    # Keep only the top X docs returned
    formatted_documents, metadata, ids, distances = formatted_documents[:n_results], metadata[:n_results], ids[:n_results], distances[:n_results]

    # Create one message per source document
    document_messages = []
    for id, source_doc in zip(ids, formatted_documents):
        document_messages.append({
            'role': 'user',
            'content': f"{id}:\n\n{source_doc}"
        })

    # Text prompts
    prompt = prompt_template_type
    question_intro = PromptTemplateType.question_intro_en if language == "en" else PromptTemplateType.question_intro_fr
    prompt_question = "\n\n" + question_intro + ": " + database_question

    prompt_message = {
        'role': 'user',
        'content': prompt + prompt_question
    }

    messages = []
    previous_questions_and_answers = []
    previous_question_documents = []

    if nb_previous_questions > 0 and previous_messages and previous_question_chunks and len(previous_messages) >= 2:
        previous_question_text = PromptTemplateType.previous_question_en if language == "en" else PromptTemplateType.previous_question_fr
        previous_answer_text = PromptTemplateType.previous_answer_en if language == "en" else PromptTemplateType.previous_answer_fr

        for document, _, id, _ in previous_question_chunks:
            previous_question_documents.append({
                'role': 'user',
                'content': f"{id} ({previous_question_text}):\n\n{document}"
            })

        # Get nb_previous_questions Q&A pairs, in reverse order
        nb_questions = 0
        nb_answers = 0
        for message in previous_messages[::-1]:
            if nb_questions >= nb_previous_questions and nb_answers >= nb_previous_questions:
                break

            if message["role"] == "user":
                previous_question = {
                    "role": "user",
                    "content": f"{previous_question_text.capitalize()}:\n\n" + message["content"]
                }
                previous_questions_and_answers.insert(0, previous_question)
                nb_questions += 1
            else:
                previous_answer = {
                    "role": "assistant",
                    "content": f"{previous_answer_text.capitalize()}:\n\n" + message.get("original_answer")
                }
                previous_questions_and_answers.insert(0, previous_answer)
                nb_answers += 1

    num_ctx = hyperparams.get("num_ctx")

    # Remove chunks from the context if it's too long
    final_document_messages, num_ctx, total_used_tokens = manage_max_context_length(
        num_ctx, document_messages, prompt, prompt_question, previous_question_documents, 
        previous_questions_and_answers, max_context_length
    )

    # Only update the num_ctx if it's not None (always the case for vLLM)
    if num_ctx is not None:
        hyperparams["num_ctx"] = num_ctx

    # List of all messages to be sent to the LLM
    messages = final_document_messages + previous_questions_and_answers + [prompt_message]

    formatted_for_metadata_tab = format_for_metadata_tab_ui(ids, metadata)
    formatted_for_documents_tab = format_for_documents_tab_ui(ids, metadata, documents)

    chunks = [(doc, meta.get("hyperlink"), id, meta.get("title")) for doc, meta, id in zip(formatted_documents, metadata, ids)]
    
    return messages, chunks, formatted_for_metadata_tab, formatted_for_documents_tab, chat_model, hyperparams, total_used_tokens

def get_previous_question_chunks(previous_messages, nb_previous_questions=1):
    if nb_previous_questions <= 0 or not previous_messages:
        return []

    previous_question_chunks = []
    nb_prev_assistant_messages = 0
    # Get chunks from the last nb_previous_questions assistant messagesm starting from the last
    for message in previous_messages[::-1]:
        if nb_prev_assistant_messages >= nb_previous_questions:
            break

        if message["role"] == "user":
            continue

        chunks = message.get("chunks", [])
        # Add the chunks to the previous question chunks, at the beginning of the list
        for chunk in chunks:
            previous_question_chunks.insert(0, chunk)

        nb_prev_assistant_messages += 1

    return previous_question_chunks

def print_context_info(total_used_tokens, hyperparams):
    if ConsoleConfig.verbose:
        if total_used_tokens is not None:
            print(f"Total used tokens: {total_used_tokens}")
        num_ctx = hyperparams.get("num_ctx")
        if num_ctx is not None:
            print(f"Context window size: {num_ctx}")

@st.cache_data(show_spinner=False, ttl=600)
def retrieve_database_local(database_question,
                      language,
                      db_name,
                      is_remote=False,
                      chat_model=None,
                      n_results=ChromaDBSettings.nb_docs_returned,
                      hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig,
                      quotations_mode=QuotationsConfig.direct_quotations_mode,
                      custom_system_prompt=None,
                      previous_messages=None,
                      nb_previous_questions=1,
                      engine="ollama"):
    
    previous_question_chunks = get_previous_question_chunks(previous_messages, nb_previous_questions)
    
    messages, chunks, metadata_tab, documents_tab, chat_model, hyperparams, total_used_tokens = retrieve_database(
        database_question, language, db_name, is_remote, chat_model, n_results, hyperparams, quotations_mode, 
        custom_system_prompt, previous_messages, previous_question_chunks, nb_previous_questions
    )

    if is_remote:
        original_answer = get_llm_answer_remote(chat_model, messages, hyperparams)
    elif engine=="ollama":
        original_answer = get_ollama_answer_local(chat_model, messages, hyperparams)
    elif engine=="vllm":
        original_answer = get_vllm_answer(chat_model, messages, hyperparams)

    formatted_llm_answer = markdown_post_processing(original_answer)
    if not previous_question_chunks:
        previous_question_chunks = []

    formatted_llm_answer = post_processing(formatted_llm_answer, previous_question_chunks + chunks, quotations_mode)

    print_context_info(total_used_tokens, hyperparams)

    return formatted_llm_answer, metadata_tab, documents_tab, chunks, original_answer

def retrieve_database_stream(database_question,
                      language,
                      db_name,
                      is_remote=False,
                      chat_model=None,
                      n_results=ChromaDBSettings.nb_docs_returned,
                      hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig,
                      quotations_mode=QuotationsConfig.direct_quotations_mode,
                      custom_system_prompt=None,
                      previous_messages=None,
                      nb_previous_questions=1,
                      engine="ollama"):
    
    previous_question_chunks = get_previous_question_chunks(previous_messages, nb_previous_questions)

    messages, chunks, metadata_tab, documents_tab, chat_model, hyperparams, total_used_tokens = retrieve_database(
        database_question, language, db_name, is_remote, chat_model, n_results, hyperparams, quotations_mode, 
        custom_system_prompt, previous_messages, previous_question_chunks, nb_previous_questions
    )

    if is_remote:
        stream_generator = get_llm_answer_remote_stream(chat_model, messages, hyperparams)
    elif engine=="ollama":
        stream_generator = get_ollama_answer_local_stream(chat_model, messages, hyperparams)
    elif engine=="vllm":
        stream_generator = get_vllm_answer_stream(chat_model, messages, hyperparams)

    result = get_paragraph_generator(stream_generator, previous_question_chunks + chunks, quotations_mode)

    print_context_info(total_used_tokens, hyperparams)

    return result, metadata_tab, documents_tab, chunks
    

if __name__ == "__main__":
    from config import vLLMRAGConfig, vLLMChatbotInterfaceConfig

    # start_time = time.time()
    # answer, _, _, _, original_answer = retrieve_database_local(
    #     "How do you change your password in WEIMS?", 
    #     "en", 
    #     "equity",
    #     is_remote=False, 
    #     chat_model=vLLMChatbotInterfaceConfig.default_model_local,
    #     n_results=5,
    #     hyperparams=vLLMRAGConfig.HyperparametersAccuracyConfig,
    #     engine="vllm"
    # )
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")

    # #print(answer)
    # print(original_answer)

    hyperparams = vLLMRAGConfig.HyperparametersAccuracyConfig.copy()
    chat_model = vLLMChatbotInterfaceConfig.default_model_local
    
    answer, _, _, chunks = retrieve_database_stream("what is section 204 of the CLC about?", "en", "labour", chat_model=chat_model, hyperparams=hyperparams, engine="vllm", n_results=5, is_remote=False)
    previous_paragraphs = ""
    original_previous_paragraphs = ""
    for previous_paragraphs, original_previous_paragraphs, stream_generator in answer:
        print(previous_paragraphs)
        previous_paragraphs = previous_paragraphs
        original_previous_paragraphs = original_previous_paragraphs

    print(original_previous_paragraphs)
    print(previous_paragraphs)