from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # "nf4" (NormalFloat4) or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bfloat16 for speed
    bnb_4bit_use_double_quant=True      # nested quantization for extra memory savings
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"   # automatically distributes across available GPUs/CPU
)