from rag.extract_ipgs import extract_ipgs_main
from rag.extract_pdf import extract_pdfs_main
from rag.extract_law import extract_law_main
from rag.extract_page import extract_pages_main

from rag.page_utils import get_tokenizer_and_limit

class RagExtractor:
    selected_tokenizer = None
    selected_token_limit = None

    def __init__(self):
        self.selected_tokenizer, self.selected_token_limit = get_tokenizer_and_limit()
        
    def extract(self, extract_type: str, data_dict: dict, db_name: str, save_html: bool = False, blacklist_dict: dict = None):
        if extract_type == "ipg":
            extract_ipgs_main(data_dict, db_name, self.selected_tokenizer, self.selected_token_limit, save_html)
        elif extract_type == "law":
            extract_law_main(data_dict, db_name, self.selected_tokenizer, self.selected_token_limit)
        elif extract_type == "page":
            extract_pages_main(data_dict, db_name, self.selected_tokenizer, self.selected_token_limit, save_html, blacklist_dict)
        elif extract_type == "pdf":
            extract_pdfs_main(data_dict, db_name, self.selected_tokenizer, self.selected_token_limit)
        else:
            raise ValueError(f"Unknown extraction type: {extract_type}")