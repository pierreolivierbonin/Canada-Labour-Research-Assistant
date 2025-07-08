import sys
sys.path.append("./")

from config import ChromaDBSettings
from tools import _load_vector_database


def fetch_documents_from_database_simplified(database_question, 
                                  language,
                                  db_name,
                                  n_results=5):

    results = _load_vector_database(language, db_name).query(query_texts=database_question,
                                            n_results=n_results,
                                            include=["metadatas", "distances", "documents"])

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    ids = [str(i) for i in results["ids"][0]]
    distances = results["distances"][0]

    return documents, metadata, ids, distances


if __name__ == "__main__":

    related_example1 = '''The Code provides for up to 17 weeks of maternity leave. 
    However, the total duration of the maternity and the parental leaves must not exceed 78 weeks when the parental leave is not shared. 
    The total duration of the maternity and the parental leaves must not exceed 86 weeks when the parental leave is shared.'''

    docs, metadata, ids, distances = fetch_documents_from_database_simplified(related_example1, language="en", db_name="Labour")
    print(f"Related example 1: {distances}")

    unrelated_example1 = '''Those voices are known as auditory hallucinations — a hallmark of psychosis. 
    When they became more frequent and insistent, he went to the Centre for Addiction and Mental Health for an assessment.'''

    docs, metadata, ids, distances = fetch_documents_from_database_simplified(unrelated_example1, language="en", db_name="Labour")
    print(f"Unrelated example 1: {distances}")

    related_example2 = '''(2) Subject to subsection (3), the members of the Board other than the Chairperson and 
    the Vice-Chairpersons are to be appointed by the Governor in Council on the recommendation of the Minister 
    after consultation by the Minister with the organizations representative of employees or employers that the 
    Minister considers appropriate, to hold office during good behaviour for terms not exceeding three years each, 
    subject to removal by the Governor in Council at any time for cause.'''

    docs, metadata, ids, distances = fetch_documents_from_database_simplified(related_example2, language="en", db_name="Labour")
    print(f"Related example 2: {distances}")


