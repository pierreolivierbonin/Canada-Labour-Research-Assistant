import pymupdf4llm
import re
import os
import shutil
from rag.page_utils import Page, chunk_text, save_to_csv
import requests

def format_header_group(group_text, tag, page_number):
    return f"#{tag}#{group_text.strip()}/{page_number}#{tag}#"

def process_pdf(file_name, file_path, url, selected_tokenizer, selected_token_limit):
    md_pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    is_consider_emphasis_subheaders = True

    processed_pages = []
    for md_page in md_pages:
        if md_page["text"] == "":
            continue
        
        page_number = md_page["metadata"]["page"]
        
        # Split the page into lines
        lines = md_page["text"].split('\n')
        processed_lines = []
        
        for line in lines:
            # Match 1 or more # followed by a space
            header_match = re.match(r'^(#+)\s', line)
            if header_match:
                # Get the number of # characters
                header_level = len(header_match.group(1))

                tag = f'h{header_level}'

                # If line with header contains **, end the header at the end of the emphasis
                formatted_line = re.sub(r'^#+\s\*\*(.*)\*\*', lambda x: format_header_group(x.group(1), tag, page_number), line)

                # If line with header had no emphasis, end it at the end of the line
                formatted_line = re.sub(r'^#+\s(.*?)$', lambda x: format_header_group(x.group(1), tag, page_number), formatted_line)

                processed_lines.append(formatted_line)
                
            # Check for Emphasis at the start of the line if is_consider_emphasis_subheaders is True
            elif is_consider_emphasis_subheaders:
                # Replace text that start with ** and end with **, ignoring non-alphanumerical characters at start
                formatted_line = re.sub(r'^[^a-zA-Z0-9]*\*\*(.*?)\*\*', lambda x: format_header_group(x.group(1), "h4", page_number), line)
                processed_lines.append(formatted_line.strip())
            else:
                processed_lines.append(line)
        
        # Join the processed lines and add to combined text
        processed_page = '\n'.join(processed_lines)
        
        # Remove all instances of 5 dots or more
        processed_page = re.sub(r'\.{5,}', '', processed_page)

        # Remove all instances of 3 or more newlines
        processed_page = re.sub(r'\n{3,}', '\n\n', processed_page)

        # Remove all instances of 2 or more tildes in a row
        processed_page = re.sub(r'~{2,}', '', processed_page)

        # Replace <br> with a space
        processed_page = re.sub(r'<br\s*/?>', ' ', processed_page)

        # Remove all instances of •
        processed_page = re.sub(r'•', '', processed_page)

        # Remove all instances of 2 or more * in a row
        processed_page = re.sub(r'\*{2,}', '', processed_page)

        # Replace all instances of more than 1 space with a single space
        processed_page = re.sub(r'\s{2,}', ' ', processed_page)

        processed_pages.append(processed_page)

    # Combine all pages with double newlines between them
    final_text = '\n\n'.join(processed_pages)

    # Create a Page object with the processed text
    file_name_without_extension = file_name.replace(".pdf", "")
    page = Page(
        id=file_name_without_extension.replace(" ", "_"),
        title=file_name_without_extension,
        url=url,
        hierarchy=[],
        url_hierarchy=[],
        linked_pages=[],
        text=final_text,
        chunks=chunk_text(final_text, selected_tokenizer, selected_token_limit),
        date_modified=""
    )

    return page

def extract_pdfs_main(pdf_dict:dict, database_name:str, selected_tokenizer, selected_token_limit:int):
    root_folder_path = "static"

    # Create the static folder if it doesn't exist (1 liner)
    os.makedirs(root_folder_path, exist_ok=True)

    for language in pdf_dict.keys():
        # Get the pdf urls from the db_config
        pdf_urls_or_paths = pdf_dict[language]
        pages = []

        # New loop to process all paths and expand folders
        expanded_pdf_paths = []
        for pdf_url_or_path in pdf_urls_or_paths:
            if pdf_url_or_path.startswith("~"):
                pdf_url_or_path = os.path.expanduser(pdf_url_or_path)

            if os.path.isdir(pdf_url_or_path):
                # If it's a folder, add all PDF files within the folder
                for root, _, files in os.walk(pdf_url_or_path):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            expanded_pdf_paths.append(os.path.join(root, file))
            else:
                # If it's a file, add it directly
                expanded_pdf_paths.append(pdf_url_or_path)

        print(f"Processing PDFs in {language}...")

        static_local_pdf_dir = os.path.join(".", "static", database_name, language)

        # Clear static folder if it exists
        if os.path.exists(static_local_pdf_dir):
            shutil.rmtree(static_local_pdf_dir)

        # Download the pdfs to the inputs folder
        for pdf_url_or_path in expanded_pdf_paths:
            pdf_filename = os.path.basename(pdf_url_or_path)

            # create the static folder if it doesn't exist
            os.makedirs(static_local_pdf_dir, exist_ok=True)

            pdf_local_file_path = os.path.join(static_local_pdf_dir, pdf_filename) # path to the static folder in the project's root directory

            if not os.path.exists(pdf_local_file_path):
                # Check if pdf_url is a local file path, if so use it as is
                if os.path.exists(pdf_url_or_path):
                    # copy the pdf to the static folder
                    shutil.copy(pdf_url_or_path, pdf_local_file_path)
                else:
                    try:
                        print(f"Downloading {pdf_url_or_path}")
                        response = requests.get(pdf_url_or_path)

                        with open(pdf_local_file_path, "wb") as f:
                            f.write(response.content)
                    except Exception as e:
                        print(f"Error downloading {pdf_url_or_path}. Validate if the url is correct, or that the local file path is valid.")
                        raise e

            pdf_hyperlink = f"/app/static/{database_name}/{language}/{pdf_filename}" # Access the static file from the browser with the app/static/... path in the url (need to include the app/ prefix, even if not present in the directory)

            print(f"Processing {pdf_filename}")
        
            page = process_pdf(pdf_filename, pdf_local_file_path, pdf_hyperlink, selected_tokenizer, selected_token_limit)
            pages.append(page)

        # Save to CSV
        save_to_csv(pages, database_name, "pdf", language, is_pdf=True)

if __name__ == "__main__":
    from db_config import VectorDBDataFiles
    from rag.extract_pdf import extract_pdfs_main
    from rag.page_utils import get_tokenizer_and_limit

    selected_tokenizer, selected_token_limit = get_tokenizer_and_limit()
    databases = VectorDBDataFiles.databases

    for db in databases:
        db_name = db["name"]
        pdfs = db.get("pdf")

        if pdfs:
            extract_pdfs_main(pdfs, db_name, selected_tokenizer, selected_token_limit)