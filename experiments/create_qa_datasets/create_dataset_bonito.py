from bonito import Bonito
from vllm import SamplingParams
from datasets import load_dataset, Dataset
import time

start_time = time.time()

# load dataset with unannotated text
#unannotated_text = load_dataset("BatsResearch/bonito-experiment", "unannotated_contract_nli", split="train", streaming=False).select(range(10))

# Create a dataset based on clc_dataset/data.csv, where the text column is the unannotated text
clc_dataset = load_dataset("csv", data_files="./clc_dataset/data.csv", split="train", streaming=False) #.select(range(10))

# Chunk the sections from the CLC into chunks of at most 750 characters
unannotated_texts = []
section_numbers = []

def chunk_text(text, max_length):
    """Split text into chunks that end on sentence punctuation (.!?)."""
    chunks = []
    words = text.split()
    current_chunk = ""
    
    for word in words:
        # Add the word to current chunk
        if current_chunk:
            current_chunk += " " + word
        else:
            current_chunk = word
        
        # Check if we've exceeded the max length and if the word ends with sentence punctuation
        if len(current_chunk) >= max_length and word.rstrip().endswith(('.', '!', '?')):
            chunks.append(current_chunk)
            current_chunk = ""
    
    # Combine the last chunk with the current chunk if it's not empty and the last chunk is less than 100 characters
    if current_chunk and len(chunks) > 0 and len(current_chunk) <= 256:
        chunks[-1] += current_chunk
    else:
        chunks.append(current_chunk)
        
    return chunks

# Loop over all unannotated text and chunk them
for i in range(len(clc_dataset)):
    section_number = clc_dataset[i]['section_number']
    text = clc_dataset[i]['text']
    
    # Chunk the text
    chunks = chunk_text(text, 512)

    # Add section header to each chunk and add to unannotated_texts
    for part_num, chunk in enumerate(chunks, 1):
        part_text = f" part {part_num}" if len(chunks) > 1 else ""
        formatted_chunk = f"Section {section_number}{part_text} of the Canada Labour Code (CLC)\n{chunk}"
        unannotated_texts.append(chunk)
        section_numbers.append(section_number) # Note : Recently added the section number, not in current dataset.

# Create a dataset from the unannotated texts, where the text column is the unannotated text
unannotated_dataset = Dataset.from_dict({"text": unannotated_texts})

# Initialize the Bonito model
bonito = Bonito("BatsResearch/bonito-v1", enforce_eager=True) # Enforce eager mode to avoid errors (Unsupported: generator)

nb_generations = 10

# Generate synthetic instruction tuning dataset
sampling_params = SamplingParams(max_tokens=512, top_p=0.95, temperature=0.5, n=nb_generations)
synthetic_dataset = bonito.generate_tasks(
    unannotated_dataset,
    context_col="text",
    task_type="mcqa", #(has issues with the generated questions, maybe avoid)
    sampling_params=sampling_params
)

new_synthetic_data = []

# Go through the dataset and group in sets of nb_generations (since n=nb_generations generations per chunk)
for i in range(0, len(synthetic_dataset), nb_generations):
    # Look for the first generation that doesn't contain 'Would you recommend' in instruction
    included_generations = []
    row = None
    prev_not_included_row = None
    
    for j in range(0, nb_generations):
        row = synthetic_dataset[i + j]
        row['section_number'] = section_numbers[i // nb_generations]

        if 'Would you recommend' not in row['instruction']:
            included_generations.append(row)

            if len(included_generations) >= 2:
                break
        else:
            prev_not_included_row = row
            
    # If one generation was valid, double it (to avoid having the 'would you recommend' one in the dataset)
    if len(included_generations) == 1:
        included_generations.append(included_generations[0])
    # If not generation was valid, just include the 'would you recommend' one, to make sure the chunk is included.
    elif len(included_generations) == 0:
        included_generations.append(prev_not_included_row)
    
    # Add the selected generation to our new data
    new_synthetic_data += included_generations

# Create a new dataset from the filtered data
filtered_dataset = Dataset.from_list(new_synthetic_data)

# SAVE ORIGINAL DATASET (keeping the original save logic)
#synthetic_dataset.save_to_disk("synthetic_dataset_clc")
filtered_dataset.to_json("experiments/eval/clc_bonito_dataset_mcqa/data.jsonl")
print("dataset saved as 'experiments/eval/clc_bonito_dataset_mcqa/data.jsonl'")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# # Open the dataset
# synthetic_dataset = load_from_disk("synthetic_dataset")
# print(synthetic_dataset)