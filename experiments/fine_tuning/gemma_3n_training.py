# REMOVE BELOW ONCE BUG 4090 FIXED UNSLOTH
    # See : https://github.com/unslothai/unsloth/issues/2923
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

if __name__ == "__main__":
    from unsloth import FastModel
    import torch

    #import os
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        dtype = torch.bfloat16, # None for auto detection
        max_seq_length = 1024, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )

    """# Gemma 3N can process Text, Vision and Audio!

    Let's first experience how Gemma 3N can handle multimodal inputs. We use Gemma 3N's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`
    """

    from transformers import TextStreamer
    # Helper function for inference
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

    """# Gemma 3N can see images!

    <img src="https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg" alt="Alt text" height="256">
    """

    sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"

    # messages = [{
    #     "role" : "user",
    #     "content": [
    #         { "type": "image", "image" : sloth_link },
    #         { "type": "text",  "text" : "Which films does this animal feature in?" }
    #     ]
    # }]
    # # You might have to wait 1 minute for Unsloth's auto compiler
    # do_gemma_3n_inference(messages, max_new_tokens = 256)

    # """Let's make a poem about sloths!"""

    messages = [{
        "role": "user",
        "content": [{ "type" : "text",
                    "text" : "Write a poem about sloths." }]
    }]
    do_gemma_3n_inference(messages)

    # """# Gemma 3N can also hear!"""

    # audio_file = "audio.mp3"

    # messages = [{
    #     "role" : "user",
    #     "content": [
    #         { "type": "audio", "audio" : audio_file },
    #         { "type": "text",  "text" : "What is this audio about?" }
    #     ]
    # }]
    # do_gemma_3n_inference(messages, max_new_tokens = 256)

    # """# Let's combine all 3 modalities together!"""

    # messages = [{
    #     "role" : "user",
    #     "content": [
    #         { "type": "audio", "audio" : audio_file },
    #         { "type": "image", "image" : sloth_link },
    #         { "type": "text",  "text" : "What is this audio and image about? "\
    #                                     "How are they related?" }
    #     ]
    # }]
    # do_gemma_3n_inference(messages, max_new_tokens = 256)

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

    """<a name="Data"></a>
    ### Data Prep
    We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:

    ```
    <bos><start_of_turn>user
    Hello!<end_of_turn>
    <start_of_turn>model
    Hey there!<end_of_turn>
    ```

    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
    """

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    """We get the first 3000 rows of the dataset"""

    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")

    """We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!"""

    from unsloth.chat_templates import standardize_data_formats
    dataset = standardize_data_formats(dataset)

    """Let's see how row 100 looks like!"""

    print(dataset[100])

    """We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one."""

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    """Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one."""

    print(dataset[100]["text"])

    """<a name="Train"></a>
    ### Train the model
    Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """

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
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 10,
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

    # """<a name="Inference"></a>
    # ### Inference
    # Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
    # """

    # from unsloth.chat_templates import get_chat_template
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = "gemma-3",
    # )
    # messages = [{
    #     "role": "user",
    #     "content": [{
    #         "type" : "text",
    #         "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    #     }]
    # }]
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt = True, # Must add for generation
    #     return_tensors = "pt",
    #     tokenize = True,
    #     return_dict = True,
    # ).to("cuda")
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens = 64, # Increase for longer outputs!
    #     # Recommended Gemma-3 settings!
    #     temperature = 1.0, top_p = 0.95, top_k = 64,
    # )
    # tokenizer.batch_decode(outputs)

    # """ You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!"""

    # messages = [{
    #     "role": "user",
    #     "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
    # }]
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt = True, # Must add for generation
    #     return_tensors = "pt",
    #     tokenize = True,
    #     return_dict = True,
    # ).to("cuda")

    # from transformers import TextStreamer
    # _ = model.generate(
    #     **inputs,
    #     max_new_tokens = 64, # Increase for longer outputs!
    #     # Recommended Gemma-3 settings!
    #     temperature = 1.0, top_p = 0.95, top_k = 64,
    #     streamer = TextStreamer(tokenizer, skip_prompt = True),
    # )

    # messages = [{
    #     "role": "user",
    #     "content": [{ "type" : "text",
    #                 "text" : "Write a poem about sloths." }]
    # }]
    # do_gemma_3n_inference(messages)

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
    
    model.save_pretrained_merged(pretrain_model_name, tokenizer)
    quantization_type = "F16"
    model.save_pretrained_gguf(pretrain_model_name, quantization_type = quantization_type)

    # Move model.{quantization_type}.gguf to model/model.{quantization_type}.gguf
    os.rename(f"{pretrain_model_name}.{quantization_type}.gguf", f"{pretrain_model_name}/{pretrain_model_name}.{quantization_type}.gguf")

    from unsloth.save import create_ollama_modelfile

    # Save Ollama modelfile
    modelfile = create_ollama_modelfile(tokenizer, pretrain_model_name)
    modelfile_location = None
    if modelfile is not None:
        modelfile_location = os.path.join(pretrain_model_name, "Modelfile")
        with open(modelfile_location, "w") as file:
            file.write(modelfile)
        pass
        print(f"Unsloth: Saved Ollama Modelfile to {modelfile_location}")

    #model.save_pretrained_merged("gemma-3N-finetune", tokenizer)
    # Save for ollama
    # model.save_pretrained_gguf(
    #     "gemma-3n-finetune",
    #     quantization_type = "F16", # For now only Q8_0, BF16, F16 supported
    # )

    # """Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

    # if False:
    #     from unsloth import FastModel
    #     model, tokenizer = FastModel.from_pretrained(
    #         model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    #         max_seq_length = 2048,
    #         load_in_4bit = True,
    #     )

    # messages = [{
    #     "role": "user",
    #     "content": [{"type" : "text", "text" : "What is Gemma-3N?",}]
    # }]
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt = True, # Must add for generation
    #     return_tensors = "pt",
    #     tokenize = True,
    #     return_dict = True,
    # ).to("cuda")

    # from transformers import TextStreamer
    # _ = model.generate(
    #     **inputs,
    #     max_new_tokens = 128, # Increase for longer outputs!
    #     # Recommended Gemma-3 settings!
    #     temperature = 1.0, top_p = 0.95, top_k = 64,
    #     streamer = TextStreamer(tokenizer, skip_prompt = True),
    # )

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