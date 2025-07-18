from unsloth import FastLanguageModel
import torch

max_seq_length = None #2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING # "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [                    # Change below!
    #{"role": "user", "content": "Can you describe section 204 of the Canada labour code?"},
    # {"role": "user",      "content": "Continue the fibonacci sequence! Your input is 1, 1, 2, 3, 5, 8"},
    {     "role": "user",
                "content": "IPG-057_2:\n\n{\"id\": \"IPG-057_2\", \"type\": \"Interpretations, Policies and Guidelines\", \"title\": \"Reasonably practicable - Maternity and work-related illness/injury - IPG-057\", \"section\": \"N/A\"}, \"text\": \"BACKGROUND There is a need to communicate a national approach in interpreting and applying the term \"Reasonably Practicable\". Reassignment and job modification are provided for in section 204 which states: \"(1) An employee who is pregnant or nursing may, during the period from the beginning of the pregnancy to the end of the twenty-fourth week following the birth, request the employer to modify her job functions or reassign her to another job if, by reason of the pregnancy or nursing, continuing any of her current job functions may pose a risk to her health or to that of the foetus or child. (2) An employee's request under subsection (1) must be accompanied by a certificate from a health care practitioner of the employee's choice indicating the expected duration of the potential risk and the activities or conditions to avoid in order to eliminate the risk.\" The employer's obligations are provided for in section 205 which states: \"(1) An employer to whom a request has been made under subsection 204(1) shall examine the request in consultation with the employee and, where reasonably practicable, shall modify the employee's job functions or reassign her. (3) The onus is on the employer to show that a modification of job functions or a reassignment that would avoid the activities or conditions indicated in the certificate issued under subsection 204(2) is not reasonably practicable. (4) If the employer concludes that a modification of job functions or a reassignment that would avoid the activities or conditions indicated in the certificate is not reasonably practicable, the employer shall inform the employee in writing.\" Return to Work is provided for in subsection 239.1(3) which states: \"(3) Subject to the regulations, the employer shall, where reasonably practicable, return an employee to work after the employee's absence due to work-related illness or injury.\" The above provisions apply to student interns undertaking internships to fulfill the requirements of their educational program. The only exception is if a health care practitioner issues a certificate stating that a student intern is unable to continue performing internship activities, or if the employer informs the student intern in writing that a proposed modification or reassignment is not \u201cReasonably Practicable\u201d. In such case, it is at the discretion of the employer to give the student intern a leave of absence until they are able to resume their internship.\""
            },
    {"role": "user", "content": """Answer the question based on the previous documents.
Structure your responses with section headers and subtitles.
Use quotation marks to directly quote relevant passages from the text, giving an in depth analysis of how the quote relates to the question.
Include at least 1 quotation in your answer, ideally more.
Do not refer to examples from the source documents, unless you quote them first.
Those passages should be quoted word for word, avoiding the use of ellipsis to shorten the quote.
Avoid quoting passages inline with your text, instead quote them on a new line.
Do not quote the same passage twice.
Do not list the quotes you used at the end of your answer.

HERE IS THE QUESTION: What accommodations can an employee who is pregnant or nursing request from their employer, and what documentation is required to support such a request?"""},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 1200) #, pad_token_id = tokenizer.eos_token_id)
