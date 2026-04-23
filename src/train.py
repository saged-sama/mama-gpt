from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import torch
from models.mamma import Mamma
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler, autocast
import os

VOCAB_SIZE = 50_000
BATCH_SIZE = 16
MAX_SAMPLES = 50_000
MODEL_NAME = "mama-gpt"
MODEL_PATH = f"output/{MODEL_NAME}"
CONTEXT_LENGTH = 1024
NUM_WORKERS = 16
LR_SCHEDULER_STEP_SIZE = 10
MAX_NORM = 1.0
ACCUMULATION_STEPS = 16
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = 1e-6
WARM_UP_STEPS=1000
WARM_UP_FACTOR=2
WEIGHT_DECAY = 0.1
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 2048

print(f"Training {MODEL_NAME} with\nvocab size = {VOCAB_SIZE}\nbatch size = {BATCH_SIZE}\nmax samples = {MAX_SAMPLES}\ncontext length = {CONTEXT_LENGTH}\nmodel dimension = {d_model}\nnumber of heads = {num_heads}\nnumber of layers = {num_layers}\nfeedforward dimension = {d_ff}\n\n")

print("Loading dataset...")
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)
print("Dataset loaded. ✅")

def batch_iterator(batch_size=1000, max_samples=1_000_000):
    batch = []
    count = 0

    for example in ds:
        batch.append(example["text"])
        count += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

        if count >= max_samples:
            break

    if batch:
        yield batch

tokenizer_path = f"{MODEL_PATH}/tokenizer.json"
os.makedirs(f"{MODEL_PATH}", exist_ok=True)

if os.path.exists(tokenizer_path):
    print(f"Loading tokenizer from path: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("Tokenizer loaded successfully ✅")
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
            batch_size=BATCH_SIZE,
            max_samples=MAX_SAMPLES
        ),
        trainer=trainer
    )

    tokenizer.save(f"{MODEL_PATH}/tokenizer.json")
    print("Tokenizer trained and saved. ✅")

print(f"Total Vocab Size: {tokenizer.get_vocab_size()}\n\n")


def tokenize_fn(examples):
    """Tokenize examples in parallel across num_workers."""
    encodings = tokenizer.encode_batch(examples["text"])
    input_ids = []
    
    for enc in encodings:
        ids = enc.ids[:CONTEXT_LENGTH + 1]
        ids = ids + [0] * (CONTEXT_LENGTH + 1 - len(ids))
        input_ids.append(ids)
    
    return {
        "input_ids": input_ids
    }

# Map the dataset (this stays streaming!)
print("Tokenizing dataset with parallel workers...")
tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
tokenized_ds = tokenized_ds.with_format("torch")
print("Dataset tokenized. ✅\n")

# Use a standard DataLoader with the tokenized dataset
batch_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,  # Already parallelized in map(), set to 0
    pin_memory=True,
    prefetch_factor=2
)

def load_checkpoint(path, model, optimizer, scaler, scheduler):
    if os.path.exists(path):
        print(f"Restoring from {path}")
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        scheduler.load_state_dict(ckpt['lr_sched_state_dict'])
        return ckpt['step'], ckpt.get('min_loss', float('inf'))
    return 0, float('inf')

model = Mamma(
    vocab_size=VOCAB_SIZE,
    dim=d_model,
    context_length=CONTEXT_LENGTH,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=d_ff
)
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters. ✅\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model moved to {device} ✅:\n{torch.cuda.get_device_properties(device).name if device == 'cuda' else 'CPU'}\n")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_sched = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optimizer,
    T_0=WARM_UP_STEPS,
    T_mult=WARM_UP_FACTOR,
    eta_min=MINIMUM_LEARNING_RATE
)
scaler = GradScaler()
loss_fn = torch.nn.CrossEntropyLoss()
start_step, min_loss = load_checkpoint(
    f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt", 
    model=model, 
    optimizer=optimizer, 
    scaler=scaler, 
    scheduler=lr_sched
)

for step, batch in enumerate(batch_loader, start=start_step):
    # x is all tokens except last, y is all tokens except first
    x = batch["input_ids"][:, :-1].to(device)
    y = batch["input_ids"][:, 1:].to(device)

    with autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss = loss / ACCUMULATION_STEPS

    # Scaled backprop
    scaler.scale(loss).backward()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_sched.step()  # Step the scheduler

    lrs = lr_sched.get_last_lr()
    curr_loss = loss.item() * ACCUMULATION_STEPS
    print(f"Step {step+1}, Loss: {curr_loss:.4f}, Tokens Processed: {((step+1)*BATCH_SIZE*CONTEXT_LENGTH):,}, lr: {(sum(lrs) / len(lrs)):.4e}")
    
    if (step + 1) % 1000 == 0:
        prompt = "Today's world is heavily "
        
        print(f"Step {step+1}, generating from prompt: {prompt}")
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'lr_sched_state_dict': lr_sched.state_dict(),
            'min_loss': min(min_loss, curr_loss)
        }
        
        prompt_ids = tokenizer.encode(prompt).ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        generated_tokens = model.generate(
            x=prompt_tensor,
            max_new_tokens=40,
            temperature=0.7,
            top_k=50
        )
        
        decoded_output = tokenizer.decode(generated_tokens[0].tolist())
        print(f"Generated Text:\n{decoded_output}")
        
        torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt")
        print(f"Latest checkpoint saved at step {step+1}")
        
        if curr_loss < min_loss:
            min_loss = curr_loss
            torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_best.pt")
            print(f"Best checkpoint saved at step {step+1}, loss: {min_loss:.4f}")