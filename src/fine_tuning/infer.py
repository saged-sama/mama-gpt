from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from load_llm import model

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH = "./out/lora/qlora_llama_3"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load LoRA adapter
trained_model = PeftModel.from_pretrained(
    model,
    LORA_PATH
)

trained_model.eval()

while True:
    user_input = input("User: ")
    
    if user_input in ["quit", "exit", "q"]:
        break
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("Assistant1: ", response)
    print()
    
    with torch.no_grad():
        outputs = trained_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("Assistant2: ", response)
    print()