import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTConfig, SFTTrainer

from models.mamma import Mamma


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

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return CausalLMOutput(loss=loss, logits=logits)


# =========================================================
# TOKENIZER
# =========================================================
tokenizer = PreTrainedTokenizerFast(tokenizer_file="output/mama-gpt/tokenizer.json")

tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<EOS>"
tokenizer.bos_token = "<BOS>"


# =========================================================
# MODEL
# =========================================================
raw_model = Mamma(
    vocab_size=50000,
    dim=768,
    context_length=1024,
    num_layers=12,
    num_heads=12,
    hidden_dim=3072
)


# =========================================================
# LOAD CHECKPOINT
# =========================================================
checkpoint_path = "output/mama-gpt/checkpoint_mama-gpt_latest_sft.pt"

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

    prompt = f"""<BOS>#####
Instruction:
Summarize this article: {example['story']}

#####
Response:
"""

    full_text = prompt + example["summary"] + "<EOS>"

    prompt_ids = tokenizer(prompt, truncation=True, max_length=1024)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=1024)["input_ids"]

    labels = full_ids.copy()

    # ✅ CRITICAL FIX: mask prompt tokens
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return {
        "input_ids": full_ids,
        "labels": labels
    }


print("Processing dataset...")
formatted_train_dataset = train_dataset.map(preprocess)
# formatted_val_dataset = val_dataset.map(preprocess)


# =========================================================
# TRAINING CONFIG (CLEAN)
# =========================================================
training_args = SFTConfig(
    output_dir="./output/mama-gpt-sft",

    per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    gradient_accumulation_steps=16,

    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    # warmup_ratio=2,

    logging_steps=10,

    max_steps=200,   # 👈 better than epochs for streaming

    # eval_strategy="steps",
    # eval_steps=50,
    save_strategy="no",
    report_to="none",

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    optim="adamw_torch",

    max_length=1024,
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


# =========================================================
# SAVE
# =========================================================
torch.save(
    model.model.state_dict(),
    "output/mama-gpt/checkpoint_mama-gpt_latest_sft.pt"
)

print("Training complete ✅")