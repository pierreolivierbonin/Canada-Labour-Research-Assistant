import pymupdf4llm
import re
import os
from rag_utils.page_utils import Page, chunk_text, save_to_csv
import requests

def process_pdf(file_name, file_path, url, selected_tokenizer, selected_token_limit):
    md_pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    #first_header_level = None
    is_consider_bulletpoints_subheaders = True

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

                # # The lever of the first markdown header defines the base level of the headers
                # if first_header_level is None:
                #     first_header_level = header_level

                # Remove the # and * characters and any leading spaces
                clean_line = line.lstrip('# *').strip(" *")

                # tag = 'h1' if first_header_level == header_level else 'h3'

                tag = f'h{header_level}'
                processed_lines.append(f"#{tag}#{clean_line}/{page_number}#{tag}#")
                
            # Check for bullet points if is_consider_bulletpoints_subheaders is True
            elif is_consider_bulletpoints_subheaders:
                # Match lines that start with ** and end with **, ignoring non-alphanumerical characters at start
                bullet_match = re.match(r'^[^a-zA-Z0-9]*\*\*(.*)\*\*(.*)', line)
                if bullet_match:
                    # Extract the content between **
                    bullet_content = bullet_match.group(1)
                    after_bullet_content = bullet_match.group(2)

                    processed_lines.append(f"#h4#{bullet_content}/{page_number}#h4#")
                    if after_bullet_content:
                        processed_lines.append(after_bullet_content)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        # Join the processed lines and add to combined text
        processed_page = '\n'.join(processed_lines)

        # Clean page a bit 
        
        # Remove all instances of 5 dots or more
        processed_page = re.sub(r'\.{5,}', '', processed_page)

        # Remove all instances of 3 or more newlines
        processed_page = re.sub(r'\n{3,}', '\n\n', processed_page)

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

def extract_pdfs_main(pdf_dict, database_name, selected_tokenizer, selected_token_limit):
    root_folder_path = "inputs"

    # Create the inputs folder if it doesn't exist (1 liner)
    os.makedirs(root_folder_path, exist_ok=True)
    
    #languages = ["en", "fr"]
    
    for language in pdf_dict.keys():
        # Get the pdf urls from the db_config
        #pdf_urls = WebCrawlConfig.pdf_urls_fr if language == "fr" else WebCrawlConfig.pdf_urls
        pdf_urls = pdf_dict[language]
        pages = []

        folder_path = os.path.join(root_folder_path, language)
        os.makedirs(folder_path, exist_ok=True)

        print(f"Processing PDFs in {language}...")

        # Download the pdfs to the inputs folder
        for pdf_url in pdf_urls:
            pdf_file = pdf_url.split("/")[-1]
            pdf_file_path = os.path.join(folder_path, pdf_file)

            if not os.path.exists(pdf_file_path):
                print(f"Downloading {pdf_url}")
                response = requests.get(pdf_url)

                with open(pdf_file_path, "wb") as f:
                    f.write(response.content)

            print(f"Processing {pdf_file}")
        
            page = process_pdf(pdf_file, pdf_file_path, pdf_url, selected_tokenizer, selected_token_limit)
            pages.append(page)

        # Save to CSV
        save_to_csv(pages, database_name, "pdfs", language, is_pdf=True)

# if __name__ == "__main__":
#     main()
