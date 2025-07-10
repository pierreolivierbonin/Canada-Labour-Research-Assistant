'''
This experiment takes into account issues detailed in https://arxiv.org/pdf/2403.05440v1
'''

import sys
sys.path.append("./")

import chromadb
from chromadb.config import Settings

from config import ChromaDBSettings
from db_config import EmbeddingModel, ModelsConfig

# create client
client = chromadb.PersistentClient(path=ChromaDBSettings.directory_path,
                                   settings=Settings(anonymized_telemetry=False))

def MRE_cosine():
    print("\n\nBeginning Minimal Reproducible Example of embedding model with cosine distance...")
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["mpnet"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    # fetch or create collection
    collection = client.get_or_create_collection(name="MRE_cosine_distance",
                                                    embedding_function=selected_model.model_chroma_callable,
                                                    configuration={"hnsw": {"space": "cosine",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                            "ef_construction": 1000}                       
                                                })

    test_document_chunk = '''Section 247.6 (1) An employee who takes a leave of absence under this Division shall (a) unless there is a valid reason for not doing so, give at least four weeks’ notice to the employer before the day on which the leave is to begin; and (b) inform the employer of the length of the leave. Marginal note: If there is a valid reason (2) If there is a valid reason for not providing notice in accordance with paragraph (1)(a), the employee shall notify the employer as soon as practicable that the employee is taking a leave of absence. Marginal note: Change in length of leave (3) Unless there is a valid reason for not doing so, an employee who takes a leave of absence under this Division shall notify the employer of any change in the length of the leave at least four weeks before (a) the new day on which the leave is to end, if the employee is taking a shorter leave; or (b) the day that was most recently indicated for the leave to end, if the employee is taking a longer leave. Marginal note: In writing (4) Unless there is a valid reason for not doing so, any notice or other information to be provided by the employee to the employer under this section is to be in writing. 2008, c. 15, s. 1 Marginal note: Request for proof'''
    embeddings = selected_model.model_chroma_callable(test_document_chunk)

    collection.upsert(
        embeddings=embeddings,
        documents=test_document_chunk,
        ids=["id1"]
    )

    results = collection.query(query_texts="Rayleigh scattering is the scattering or deflection of light",          # unrelated (~=1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nCosine distance, unrelated document: {results["distances"][0][0]}")


    results = collection.query(query_texts="employees taking leave",                                                # paraphrase (>=0, <1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nCosine distance, paraphrase: {results["distances"][0][0]}")

    results = collection.query(query_texts="An employee who takes a leave of absence under this Division shall",    # direct excerpt (>=0, <1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nCosine distance, direct excerpt: {results["distances"][0][0]}")

    results = collection.query(query_texts=test_document_chunk,                                                     # identical document chunk (0)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nCosine distance, identical document: {results["distances"][0][0]}")
    assert results["distances"][0][0]==0.0


def MRE_ip():
    print("\n\nBeginning Minimal Reproducible Example of embedding model with dot product...")
    selected_model = EmbeddingModel(model_name=ModelsConfig.models["multi_qa"], trust_remote_code=True)
    selected_model.assign_model_and_attributes()

    # fetch or create collection
    collection = client.get_or_create_collection(name="MRE_cosine_distance",
                                                    embedding_function=selected_model.model_chroma_callable,
                                                    configuration={"hnsw": {"space": "ip",     # https://docs.trychroma.com/docs/collections/configure#spann-index-configuration
                                                                            "ef_construction": 1000},
                                                })

    test_document_chunk = '''Section 247.6 (1) An employee who takes a leave of absence under this Division shall (a) unless there is a valid reason for not doing so, give at least four weeks’ notice to the employer before the day on which the leave is to begin; and (b) inform the employer of the length of the leave. Marginal note: If there is a valid reason (2) If there is a valid reason for not providing notice in accordance with paragraph (1)(a), the employee shall notify the employer as soon as practicable that the employee is taking a leave of absence. Marginal note: Change in length of leave (3) Unless there is a valid reason for not doing so, an employee who takes a leave of absence under this Division shall notify the employer of any change in the length of the leave at least four weeks before (a) the new day on which the leave is to end, if the employee is taking a shorter leave; or (b) the day that was most recently indicated for the leave to end, if the employee is taking a longer leave. Marginal note: In writing (4) Unless there is a valid reason for not doing so, any notice or other information to be provided by the employee to the employer under this section is to be in writing. 2008, c. 15, s. 1 Marginal note: Request for proof'''
    embeddings = selected_model.model_chroma_callable(test_document_chunk)

    collection.upsert(
        embeddings=embeddings,
        documents=test_document_chunk,
        ids=["id1"]
    )

    results = collection.query(query_texts="Rayleigh scattering is the scattering or deflection of light",          # unrelated (~=1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nDot product, unrelated document: {results["distances"][0][0]}")


    results = collection.query(query_texts="employees taking leave",                                                # paraphrase (>=0, <1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nDot product, paraphrase: {results["distances"][0][0]}")

    results = collection.query(query_texts="An employee who takes a leave of absence under this Division shall",    # direct excerpt (>=0, <1)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nDot product, direct excerpt: {results["distances"][0][0]}")

    results = collection.query(query_texts=test_document_chunk,                                                     # identical document chunk (0)
                                n_results=1,
                                include=["metadatas", "distances", "documents"])
    print(f"\nDot product, identical document: {results["distances"][0][0]}")


if __name__ == "__main__":
    MRE_cosine()

    MRE_ip()

    # conclusion: as expected, the cosine distance is 0 for identical matches. And very close to 0 for dot product.
    # Takeaway: when creating a ChromaDB collection, make sure the embedding model's similarity function specified matches the natively-supported
    # similarity function of the embedding model. Otherwise, results may be inconsistent.