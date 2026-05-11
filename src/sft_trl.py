import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTConfig, SFTTrainer

from src.models.mamma import Mamma


MODEL_NAME="mama-gpt-larger"
CONTEXT_LENGTH=3072

# =========================================================
# CONFIG
# =========================================================
class MammaConfig(PretrainedConfig):
    model_type = "mamma"

    def __init__(self, **kwargs):
        self._attn_implementation = "sdpa"
        self.use_cache = False
        super().__init__(**kwargs)


# =========================================================
# HF WRAPPER (FIXED LOSS - NO DOUBLE SHIFT BUGS)
# =========================================================
class HFCompatibleMamma(PreTrainedModel):
    config_class = MammaConfig

    def __init__(self, base_model):
        super().__init__(MammaConfig())
        self.model = base_model
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.output

    def set_output_embeddings(self, value):
        self.model.output = value

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            # ✅ TRL/HF EXPECTS SHIFT HERE (we keep it correct & simple)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # non_masked = (labels != -100).sum().item()
            # total = labels.numel()
            # print(f"labels shape: {labels.shape}, non-masked: {non_masked}/{total}, "
            #     f"sample: {labels[0, -10:].tolist()}")

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(loss=loss, logits=logits)


# =========================================================
# TOKENIZER
# =========================================================
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"output/{MODEL_NAME}/tokenizer.json")

tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<EOS>"
tokenizer.bos_token = "<BOS>"


# =========================================================
# MODEL
# =========================================================
raw_model = Mamma(
    vocab_size=32_000,
    dim=768,
    context_length=CONTEXT_LENGTH,
    num_layers=12,
    num_heads=12,
    hidden_dim=int(8*768/3)
)


# =========================================================
# LOAD CHECKPOINT
# =========================================================
checkpoint_path = f"output/{MODEL_NAME}/checkpoint_{MODEL_NAME}_latest.pt"

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        raw_model.load_state_dict(ckpt["model_state_dict"])
    else:
        raw_model.load_state_dict(ckpt)

    print("Checkpoint loaded ✅")
else:
    print("No checkpoint found — training from scratch")


model = HFCompatibleMamma(raw_model)

model.model.train()


# =========================================================
# DATASET (FIXED + PROPER MASKING)
# =========================================================
dataset = load_dataset(
    "glnmario/news-qa-summarization",
    streaming=True
)

train_dataset = dataset["train"]
# val_dataset = dataset["validation"]

def preprocess(example):
    # system = example.get("system", "")
    
    story: str = example['story']
    
    clean_story = story.split(" -- ", 1)[-1]

    prompt = (
        f"<context>{clean_story}</context>\n"
        f"<summary>"
    )

    summary_ids = tokenizer(example["summary"] + "</summary><EOS>",
                        truncation=False)["input_ids"]
    prompt_ids  = tokenizer(prompt, truncation=False)["input_ids"]

    # Truncate prompt (not the completion) if needed
    max_prompt = CONTEXT_LENGTH - len(summary_ids)
    prompt_ids = prompt_ids[:max_prompt]

    full_ids = prompt_ids + summary_ids
    labels = [-100] * len(prompt_ids) + summary_ids.copy()

    # Hard cap just in case
    full_ids = full_ids[:CONTEXT_LENGTH]
    labels   = labels[:CONTEXT_LENGTH]
    return {
        "input_ids": full_ids,
        "labels": labels
    }


print("Processing dataset...")
# train_dataset = train_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
# val_dataset = val_dataset.filter(lambda x: len(x["answers"]["text"]) > 0)
formatted_train_dataset = train_dataset.map(preprocess)
# formatted_val_dataset = val_dataset.map(preprocess)

# sample = next(iter(formatted_val_dataset))
# answer_ids = [l for l in sample["labels"] if l != -100]
# prompt_mask_count = len([l for l in sample["labels"] if l == -100])

# print(f"Prompt tokens (masked): {prompt_mask_count}")
# print(f"Answer token ids: {answer_ids}")
# print(f"Answer decoded: {tokenizer.decode(answer_ids)}")
# print(f"Original answer: {next(iter(val_dataset.filter(lambda x: len(x['answers']['text']) > 0)))['answers']['text'][0]}")

# =========================================================
# TRAINING CONFIG (CLEAN)
# =========================================================
training_args = SFTConfig(
    output_dir=f"./output/{MODEL_NAME}-sft",

    per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    gradient_accumulation_steps=16,

    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    # warmup_ratio=2,

    logging_steps=10,

    max_steps=1300,   # 👈 better than epochs for streaming

    # eval_strategy="steps",
    # eval_steps=500,
    save_strategy="no",
    report_to="none",

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    optim="adamw_8bit",

    max_length=CONTEXT_LENGTH,
    gradient_checkpointing=False,

    # ✅ IMPORTANT FOR STABILITY
    packing=False,
    dataset_text_field=None,
)


# =========================================================
# TRAINER
# =========================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train_dataset,
    # eval_dataset=formatted_val_dataset,
    processing_class=tokenizer,
)


# =========================================================
# TRAIN
# =========================================================
print("Starting SFT training...")
trainer.train()

model.model.eval()

# =========================================================
# SAVE
# =========================================================
torch.save(
    model.model.state_dict(),
    f"output/{MODEL_NAME}/checkpoint_{MODEL_NAME}_latest_sft.pt"
)

print("Training complete ✅")