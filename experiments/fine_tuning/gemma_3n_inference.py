import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from unsloth import FastLanguageModel
import torch

max_seq_length = None #2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "gemma-3n-lora", # YOUR MODEL YOU USED FOR TRAINING # "unsloth/Llama-3.2-3B-Instruct",
    dtype = torch.bfloat16, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [                    # Change below!
    {"role": "user", "content": [{ "type" : "text", "text" : "Write a poem about sloths." }]}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)

# Convert inputs to bfloat16 to match model dtype (occurs when using sound as inputs, float32 by default it seems)
for key in inputs:
    if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype in [torch.float32, torch.float16]:
        inputs[key] = inputs[key].to(torch.bfloat16)

_ = model.generate(
    **inputs,
    max_new_tokens = 128,
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = text_streamer,
)