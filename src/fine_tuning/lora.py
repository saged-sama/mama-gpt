from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from src.fine_tuning.load_llm import model, tokenizer
from trl import SFTTrainer, SFTConfig

import matplotlib.pyplot as plt

tokenizer.pad_token = "<|finetune_right_pad_id|>" 

# =========================================================
# MEMORY OPTIMIZATION
# =========================================================

model.gradient_checkpointing_enable()

# =========================================================
# LOAD DATASETS
# =========================================================

mbpp_ds = load_dataset(
    "Muennighoff/mbpp",
    # streaming=True,
    split="test"
)

small_code_alpaca = load_dataset(
    "sahil2801/CodeAlpaca-20k",
    # streaming=True,
    split="train"
)

human_eval = load_dataset(
    "openai/openai_humaneval",
    # streaming=True,
    split="test"
)

caiss_mmlu = load_dataset(
    "cais/mmlu",
    "college_computer_science",
    # streaming=True,
    split="validation"
)

# =========================================================
# GENERIC PREPROCESSOR
# =========================================================
def preprocess(prompt, answer):
    formatted_text = (
        "System: You are a helpful assistant\n\n"
        f"User: {prompt}\n\n"
        f"Assistant: ```\n{answer}\n```"
    )

    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt"
    )

    instruction_text = (
        "System: You are a helpful assistant\n\n"
        f"User: {prompt}\n\n"
        "Assistant: ```\n"
    )

    instruction_len = len(tokenizer(
        instruction_text,
        truncation=True,
        max_length=2048,
        add_special_tokens=False   # ← avoid BOS mismatch
    )["input_ids"])

    input_ids = tokenized["input_ids"][0].tolist()
    attention_mask = tokenized["attention_mask"][0].tolist()

    labels = [-100] * len(input_ids)

    # Only unmask actual answer tokens (not padding)
    for i in range(instruction_len, len(input_ids)):
        if attention_mask[i] == 1:   # ← mask out padding
            labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# =========================================================
# MBPP FORMATTER
# =========================================================

def format_mbpp(example):
    return preprocess(
        prompt=example["text"],
        answer=example["code"]
    )

# =========================================================
# CODE ALPACA FORMATTER
# =========================================================

def format_code_alpaca(example):
    prompt = example["instruction"]

    if example["input"] != "":
        prompt += f"\n\n{example['input']}"

    return preprocess(
        prompt=prompt,
        answer=example["output"]
    )

# =========================================================
# HUMAN EVAL FORMATTER
# =========================================================

def format_human_eval(example):
    return preprocess(
        prompt=example["prompt"],
        answer=example["canonical_solution"]
    )

# =========================================================
# MMLU FORMATTER
# =========================================================

def format_mmlu(example):

    choices = "\n".join([
        f"{chr(65+i)}. {choice}"
        for i, choice in enumerate(example["choices"])
    ])

    prompt = (
        f"{example['question']}\n\n"
        f"{choices}\n\n"
        "Answer:"
    )

    answer = chr(65 + example["answer"])

    return preprocess(
        prompt=prompt,
        answer=answer
    )

# =========================================================
# FORMAT DATASETS
# =========================================================

formatted_mbpp = mbpp_ds.map(format_mbpp)

formatted_code_alpaca = small_code_alpaca.map(format_code_alpaca)

formatted_train_dataset = concatenate_datasets([
    formatted_mbpp,
    formatted_code_alpaca
])

formatted_human_eval = human_eval.map(format_human_eval)

formatted_mmlu = caiss_mmlu.map(format_mmlu)

formatted_val_dataset = concatenate_datasets([
    formatted_human_eval,
    formatted_mmlu
])

# =========================================================
# LORA CONFIG
# =========================================================

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    bias="none"
)

# =========================================================
# TEST PROMPTS
# =========================================================

inputs = [
    {
        "input": tokenizer(
            "The name of the mother of computer programming is",
            return_tensors="pt"
        ).to("cuda"),
        "max_new_tokens": 6
    },
    {
        "input": tokenizer(
            "2+2=",
            return_tensors="pt"
        ).to("cuda"),
        "max_new_tokens": 1
    },
    {
        "input": tokenizer(
            "A simple cpp function to add 2 numbers:\n\n```cpp\nint sum(int a, int b){\n",
            return_tensors="pt"
        ).to("cuda"),
        "max_new_tokens": 20
    },
]

def get_output(input):
    outputs = model.generate(
        **input["input"],
        max_new_tokens=input["max_new_tokens"]
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

print("Base Model Outputs:\n")

for input in inputs:
    print(get_output(input), "\n")

# =========================================================
# APPLY LORA
# =========================================================

model = get_peft_model(model, lora_config)

# =========================================================
# TRAIN CONFIG
# =========================================================

train_config = SFTConfig(
    output_dir="./out/lora/",

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,

    learning_rate=2e-5,
    num_train_epochs=2,

    lr_scheduler_type="cosine",

    optim="paged_adamw_8bit",

    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,

    logging_strategy="steps",
    logging_steps=10,

    eval_strategy="steps",
    eval_steps=50,

    max_grad_norm=0.3,
    warmup_ratio=0.03,

    bf16=True,

    report_to="none"
)

# =========================================================
# TRAINER
# =========================================================

trainer = SFTTrainer(
    model=model,
    args=train_config,

    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_val_dataset,

    processing_class=tokenizer,
)

# =========================================================
# TRAIN
# =========================================================

model.eval()
print(trainer.evaluate())

model.train()
trainer.train()

# =========================================================
# SAVE MODEL
# =========================================================

model.save_pretrained("./out/lora/qlora_llama_3")
tokenizer.save_pretrained("./out/lora/qlora_llama_3")

# =========================================================
# POST TRAIN EVAL
# =========================================================

model.eval()
trainer.evaluate()

print("\nFine Tuned Outputs:\n")

for input in inputs:
    print(get_output(input), "\n")

# =========================================================
# PLOT LOSSES
# =========================================================

train_losses = []
eval_losses = []

train_steps = []
eval_steps = []

for log in trainer.state.log_history:

    if "loss" in log:
        train_losses.append(log["loss"])
        train_steps.append(log["step"])

    if "eval_loss" in log:
        eval_losses.append(log["eval_loss"])
        eval_steps.append(log["step"])

plt.figure(figsize=(10, 5))

plt.plot(train_steps, train_losses, label="Train Loss")
plt.plot(eval_steps, eval_losses, label="Validation Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.savefig("./out/lora/loss_curve.png")

print("\nSaved loss curve to ./out/lora/loss_curve.png")