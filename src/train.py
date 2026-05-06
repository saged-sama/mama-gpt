from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast

import random
import os
import math

from models.mamma import Mamma

# =========================================================
# CONFIG
# =========================================================

VOCAB_SIZE = 32_000
BATCH_SIZE = 8
TOKENIZER_SAMPLES = 20_000
MAX_TRAINING_STEPS = 150_000
MODEL_NAME = "mama-gpt-larger"
MODEL_PATH = f"output/{MODEL_NAME}"

CONTEXT_LENGTH = 3072
NUM_WORKERS = 0

ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = LEARNING_RATE * 0.1
WARM_UP_STEPS = 2000
WEIGHT_DECAY = 0.1
MAX_NORM = 1.0

d_model = 768
num_heads = 12
num_layers = 12
d_ff = int(8 * d_model / 3)

os.makedirs(MODEL_PATH, exist_ok=True)

print(f"""
Training {MODEL_NAME}

vocab size       = {VOCAB_SIZE}
batch size       = {BATCH_SIZE}
context length   = {CONTEXT_LENGTH}
training steps   = {MAX_TRAINING_STEPS}
accum steps      = {ACCUMULATION_STEPS}

d_model          = {d_model}
num_heads        = {num_heads}
num_layers       = {num_layers}
d_ff             = {d_ff}
""")

# =========================================================
# DATASET MIX
# =========================================================

print("Loading datasets...")

fw = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "fineweb-edu-dedup",
    split="train",
    streaming=True
)

cosmo = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    split="train",
    streaming=True
)

print("Datasets loaded. ✅")

DATASETS = [
    ("fineweb", fw, 0.80),
    ("cosmo", cosmo, 0.20),
]


def mixed_text_stream():
    iters = {name: iter(ds) for name, ds, weight in DATASETS}

    while True:
        r = random.random()
        cumulative = 0.0

        for name, ds, weight in DATASETS:
            cumulative += weight
            if r < cumulative:
                try:
                    yield next(iters[name])["text"]
                except StopIteration:
                    # Restart exhausted iterator
                    iters[name] = iter(ds)
                    yield next(iters[name])["text"]
                break


# =========================================================
# TOKENIZER
# =========================================================

tokenizer_path = f"{MODEL_PATH}/tokenizer.json"


def batch_iterator(batch_size=1000, max_samples=20_000):
    batch = []
    count = 0

    for text in mixed_text_stream():
        batch.append(text)
        count += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

        if count >= max_samples:
            break

    if batch:
        yield batch


if os.path.exists(tokenizer_path):
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded. ✅")
else:
    print("Training tokenizer...")

    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    tokenizer.train_from_iterator(
        batch_iterator(
            batch_size=1000,
            max_samples=TOKENIZER_SAMPLES
        ),
        trainer=trainer
    )

    tokenizer.save(tokenizer_path)
    print("Tokenizer trained and saved. ✅")

bos_id = tokenizer.token_to_id("<BOS>")
eos_id = tokenizer.token_to_id("<EOS>")
print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")


# =========================================================
# TOKENIZED STREAMING DATASET
# =========================================================

class PackedDataset(IterableDataset):
    def __iter__(self):
        buffer = []

        for text in mixed_text_stream():
            ids = tokenizer.encode(text).ids

            # add explicit document boundaries
            ids = [bos_id] + ids + [eos_id]

            buffer.extend(ids)

            while len(buffer) >= CONTEXT_LENGTH + 1:
                chunk = buffer[:CONTEXT_LENGTH + 1]
                buffer = buffer[CONTEXT_LENGTH + 1:]

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long)
                }


train_ds = PackedDataset()

batch_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================

model = Mamma(
    vocab_size=VOCAB_SIZE,
    dim=d_model,
    context_length=CONTEXT_LENGTH,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=d_ff
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Using device: {device}")

# Autocast dtype: bfloat16 on CUDA, float32 on CPU
autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32


# =========================================================
# OPTIMIZER / SCHEDULER
# =========================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
    eps=1e-5
)


def lr_lambda(current_step):
    if current_step < WARM_UP_STEPS:
        return current_step / max(1, WARM_UP_STEPS)

    progress = (current_step - WARM_UP_STEPS) / max(
        1,
        MAX_TRAINING_STEPS - WARM_UP_STEPS
    )

    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    min_ratio = MINIMUM_LEARNING_RATE / LEARNING_RATE
    return min_ratio + (1 - min_ratio) * cosine_decay


lr_sched = LambdaLR(optimizer, lr_lambda)

loss_fn = torch.nn.CrossEntropyLoss()

# =========================================================
# TRAIN
# =========================================================

optimizer.zero_grad()
optimizer_step = 0
raw_step = 0

# checkpoint_path = f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt"
# chkpt = torch.load(checkpoint_path)

# model.load_state_dict(chkpt['model_state_dict'])
# optimizer.load_state_dict(chkpt['optimizer_state_dict'])
# lr_sched.load_state_dict(chkpt['lr_sched_state_dict'])
# optimizer_step = chkpt.get('optimizer_step', 0)
# raw_step = chkpt.get('raw_step', 0)
step = raw_step

for step, batch in enumerate(batch_loader, start=step):
    if optimizer_step >= MAX_TRAINING_STEPS:
        break

    x = batch["input_ids"][:, :-1].to(device)
    y = batch["input_ids"][:, 1:].to(device)

    with autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )
        scaled_loss = loss / ACCUMULATION_STEPS

    scaled_loss.backward()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad()

        optimizer_step += 1

        tokens_processed = (
            optimizer_step
            * BATCH_SIZE
            * ACCUMULATION_STEPS
            * CONTEXT_LENGTH
        )

        current_lr = lr_sched.get_last_lr()[0]

        print(
            f"Step {optimizer_step:>6} | "
            f"Loss {loss.item():.4f} | "
            f"Tokens {tokens_processed:,} | "
            f"LR {current_lr:.4e}"
        )
        
        raw_step = step

        # =================================================
        # SAMPLE GENERATION (every 10 steps)
        # =================================================
        if optimizer_step % 250 == 0:
            prompt = "It's two hearts living"

            prompt_ids = tokenizer.encode(prompt).ids
            prompt_tensor = torch.tensor(
                [prompt_ids],
                dtype=torch.long
            ).to(device)

            generated = model.generate(
                x=prompt_tensor,
                max_new_tokens=40,
                temperature=1.0,
                top_k=None,
                # eos_token_id=eos_id
            )

            text = tokenizer.decode(generated[0].tolist())

            print("\n--- SAMPLE ---")
            print(text)
            print("--------------\n")

        # =================================================
        # CHECKPOINT SAVE (every 500 steps)
        # =================================================
        if optimizer_step % 500 == 0:
            checkpoint = {
                "optimizer_step": optimizer_step,
                "raw_step": raw_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_sched_state_dict": lr_sched.state_dict(),
            }

            torch.save(
                checkpoint,
                f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt"
            )

print("Training finished.")