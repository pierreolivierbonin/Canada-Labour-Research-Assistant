# REMOVE BELOW ONCE BUG 4090 FIXED UNSLOTH
    # See : https://github.com/unslothai/unsloth/issues/2923
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

if __name__ == "__main__":
    from unsloth import FastModel
    import torch

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        # Pretrained models
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

        # Other Gemma 3 quants
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3n-E4B-it",
        dtype = None, # None for auto detection
        max_seq_length = 3072, # Length of document messages can be up to 20k characters, nan if max seq length too small.
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )

    """# Gemma 3N can process Text, Vision and Audio!

    Let's first experience how Gemma 3N can handle multimodal inputs. We use Gemma 3N's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`
    """

    from transformers import TextStreamer
    # Helper function for inference (unused for now)
    def do_gemma_3n_inference(messages, max_new_tokens = 128):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
            tokenize = True,
            return_dict = True,
        ).to("cuda")
        
        # Convert inputs to bfloat16 to match model dtype (occurs when using sound as inputs, float32 by default it seems)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype in [torch.float32, torch.float16]:
                inputs[key] = inputs[key].to(torch.bfloat16)

        _ = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature = 1.0, top_p = 0.95, top_k = 64,
            streamer = TextStreamer(tokenizer, skip_prompt = True),
        )


    """# Let's finetune Gemma 3N!

    You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!

    We now add LoRA adapters so we only need to update a small amount of parameters!
    """

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )


    import json
    questions_answers = []

    # Read the file final_questions_answers.json as json
    with open("experiments/create_qa_datasets/gemma3n_questions_answers.json", "r") as f:
        questions_answers = json.load(f)

    conversations = []

    longest_total_doc_length = 0

    for x, qa_obj in enumerate(questions_answers):
        current_conversation = []

        document_messages = qa_obj["document_messages"]
        document_ids = []
        for message in document_messages:
            current_conversation.append(message)

            document_id = message["content"].split(":\n\n")[0]
            document_ids.append(document_id)

            # For each document messae, add an assistant message that says "Document X received"
            current_conversation.append({
                "role": "assistant",
                "content": f"Document {document_id} received"
            })

        total_doc_length = sum(len(message["content"]) for message in document_messages)

        if total_doc_length > longest_total_doc_length:
            longest_total_doc_length = total_doc_length

        if total_doc_length > 10000:
            print(f"Document {document_ids} is too long for question {x}: {total_doc_length} characters")
            continue

        question = qa_obj["question"]

        current_conversation.append({
            "role": "user",
            "content": question
        })

        answer = qa_obj["answer"]

        current_conversation.append({
            "role": "assistant",
            "content": answer
        })

        conversations.append({"conversations": current_conversation})

    print(f"Longest total document length: {longest_total_doc_length}")

    from datasets import Dataset
    conversations_dataset = Dataset.from_list(conversations)

    from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template
    print(list(CHAT_TEMPLATES.keys()))

  # Supports zephyr, chatml, mistral, alpaca, vicuna, unsloth
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3", #(modified_gemma3n_template, orig_chat_template[1]), # llama = llama 2 template.
        #mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}#, # ShareGPT style
        map_eos_token = False, # Gemma3 chat templates sets it to False
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = conversations_dataset.map(formatting_prompts_func, batched = True,)

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 10,
            num_train_epochs = 2,
            # max_steps = 10,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    """We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    """Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!"""

    tokenizer.decode(trainer.train_dataset[100]["input_ids"])

    """Now let's print the masked out example - you should see only the answer is present:"""

    tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    """# Let's train the model!

    To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
    """

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    """<a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
    """

    model.save_pretrained("gemma-3n-lora")  # Local saving
    tokenizer.save_pretrained("gemma-3n-lora")
    # model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
    # tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving

    #model.save_pretrained_merged("gemma-3n-finetune", tokenizer, save_method = "merged_16bit",) 
    pretrain_model_name = "gemma3n-finetune"
    
    save_method = "merged_16bit" # Recommended to merge in 16 bit before quantization to avoid accuracy loss
    quantization_type = "Q8_0" # 4 bit not supported for gguf conversion
    model.save_pretrained_merged(pretrain_model_name, tokenizer, save_method = save_method) # Merge the model in 16 bit (necessary to save to GGUF afterwards)
    model.save_pretrained_gguf(pretrain_model_name, quantization_type = quantization_type) # Save the model in GGUF format (needed for ollama)

    # Move model.{quantization_type}.gguf to model/model.{quantization_type}.gguf
    os.rename(f"{pretrain_model_name}.{quantization_type}.gguf", f"{pretrain_model_name}/{pretrain_model_name}.{quantization_type}.gguf")

    from unsloth.save import create_ollama_modelfile

    # Save Ollama modelfile (doesn'T automatically save it otherwise for gemma-3n)
    modelfile = create_ollama_modelfile(tokenizer, pretrain_model_name)
    modelfile_location = None
    if modelfile is not None:
        # Modify the first line that starts with "FROM"
        lines = modelfile.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('FROM'):
                lines[i] = f'FROM {pretrain_model_name}.{quantization_type}.gguf'
                break
        modelfile = '\n'.join(lines)
        
        modelfile_location = os.path.join(pretrain_model_name, "Modelfile")
        with open(modelfile_location, "w") as file:
            file.write(modelfile)
        pass
        print(f"Unsloth: Saved Ollama Modelfile to {modelfile_location}")

    """### Saving to float16 for VLLM

    We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3N-finetune`. Set `if False` to `if True` to let it run!
    """

    if False: # Change to True to save finetune!
        model.save_pretrained_merged("gemma-3N-finetune", tokenizer)

    """If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

    if False: # Change to True to upload finetune
        model.push_to_hub_merged(
            "HF_ACCOUNT/gemma-3N-finetune", tokenizer,
            token = "hf_..."
        )

    """### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!
    """

    if False: # Change to True to save to GGUF
        model.save_pretrained_gguf(
            "gemma-3N-finetune",
            quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
        )

    """Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!"""

    if False: # Change to True to upload GGUF
        model.push_to_hub_gguf(
            "gemma-3N-finetune",
            quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
            repo_id = "HF_ACCOUNT/gemma-3N-finetune-gguf",
            token = "hf_...",
        )

    """Now, use the `gemma-3N-finetune.gguf` file or `gemma-3N-finetune-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    Some other links:
    1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

    <div class="align-center">
    <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
    <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

    Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
    </div>

    """