#!/usr/bin/env bash
source .venv/bin/activate

export VLLM_API_BASE="http://localhost:8000/v1"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"               # enter the value matching your GPU's compute capability, as per: https://developer.nvidia.com/cuda-gpus
export VLLM_LOGGING_LEVEL=DEBUG                     # to troubleshoot
export VLLM_ATTENTION_BACKEND=FLASHINFER            # requirements: https://docs.flashinfer.ai/installation.html
export DO_NOT_TRACK=1                               # https://docs.vllm.ai/en/v0.7.2/serving/usage_stats.html 
# export HF_HUB_OFFLINE=1                           # uncomment once you have launched the app once (and therefore downloaded the model)

# The following parameters have been tested on an RTX 4080 Super (16GB). Tweak according to available resources.
# --generation-config vllm: see the warnings here (as of 2025-05-26)--> https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
# --max-num-batched-tokens vs. max-model-len: see the explanation here--> https://github.com/vllm-project/vllm/issues/4044
# --chat-template chat_template_llama3.2_json.jinja allows to reduce the number of external dependencies (no need to pull a chat template from an external provider)
# run scripts/int4_quantization.py to quantize your model, then add '-W4A16-G128' suffix to the model name. Add --quantization compressed-tensors and serve.

# fetch the engine arguments specified in the config file
engine_args=$(python3 -c 'from config import vLLMRAGConfig; print(vLLMRAGConfig.EngineArgs["model_name"]); \
                                                            print(vLLMRAGConfig.EngineArgs["ctx_window"]); \
                                                            print(vLLMRAGConfig.EngineArgs["gpu_memory_utilization"]); \
                                                            print(vLLMRAGConfig.EngineArgs["max_num_batched_tokens"]); \
                                                            print(vLLMRAGConfig.EngineArgs["max_num_seqs"])')
params=($engine_args)
model_name="${params[0]}"
ctx="${params[1]}"
gpu_memory_utilization="${params[2]}"
max_num_batched_tokens="${params[3]}"
max_num_seqs="${params[4]}"
echo -e "\nServed model ===>" "${model_name[0]}"

# serve 
vllm serve $model_name \
    --gpu_memory_utilization $gpu_memory_utilization \
    --max-model-len=$ctx \
    --max-num-batched-tokens=$max_num_batched_tokens \
    --max-num-seqs=$max_num_seqs \
    --config config.yaml \
    --enable-prefix-caching \
    --enable-sleep-mode \
    --generation-config vllm & 

streamlit run ./chatbot_app.py vllm