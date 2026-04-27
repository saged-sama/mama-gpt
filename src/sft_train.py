from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import torch
from models.mamma import Mamma
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import os

VOCAB_SIZE = 50_000
BATCH_SIZE = 16
MAX_SAMPLES = 50_000
MODEL_NAME = "mama-gpt"
MODEL_PATH = f"output/{MODEL_NAME}"
CONTEXT_LENGTH = 1024
NUM_WORKERS = 16
MAX_NORM = 1.0
ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 3072

print(f"Training {MODEL_NAME} with\nvocab size = {VOCAB_SIZE}\nbatch size = {BATCH_SIZE}\nmax samples = {MAX_SAMPLES}\ncontext length = {CONTEXT_LENGTH}\nmodel dimension = {d_model}\nnumber of heads = {num_heads}\nnumber of layers = {num_layers}\nfeedforward dimension = {d_ff}\n\n")

print("Loading dataset...")
ds = load_dataset(
    "imone/OpenOrca_FLAN",
    split="train",
    streaming=True
)
print("Dataset loaded. ✅")

def batch_iterator(batch_size=1000, max_samples=10_000):
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

PAD_ID = tokenizer.token_to_id("<PAD>")

def format_prompt(example):
    system_prompt = example.get("system", "")
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    
    # Use the special token name you defined during tokenizer training
    eos_token = "<EOS>" 
    
    prompt = ""
    if system_prompt and system_prompt.strip():
        prompt = f"#####\nSystem:\n{system_prompt}\n\n"
        
    prompt += f"#####\nInstruction:\n{instruction}\n\n#####\nResponse:\n{response}{eos_token}"
    return prompt

def tokenize_sft_fn(example):
    full_text = format_prompt(example)
    encodings = tokenizer.encode(full_text)
    ids = encodings.ids
    
    # Truncate if over context length
    if len(ids) > CONTEXT_LENGTH:
        ids = ids[:CONTEXT_LENGTH]
        
    labels = list(ids)
    
    # 1. Mask the prompt
    response_marker = "Response:"
    marker_ids = tokenizer.encode(response_marker).ids
    
    try:
        # Find where "Response:" ends
        for i in range(len(ids) - len(marker_ids)):
            if ids[i : i + len(marker_ids)] == marker_ids:
                response_start_idx = i + len(marker_ids)
                # Mask everything before the actual response text
                for j in range(response_start_idx):
                    labels[j] = -100
                break
    except Exception:
        pass # If marker not found, it'll just train on everything

    # 2. Add Padding
    padding_len = CONTEXT_LENGTH - len(ids)
    # input_ids get PAD_ID, labels get -100 (ignored by CrossEntropy)
    input_ids = ids + [PAD_ID] * padding_len
    labels = labels + [-100] * padding_len
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# Apply this to your streaming dataset
tokenized_ds = ds.map(tokenize_sft_fn, remove_columns=ds.column_names)
# Map the dataset (this stays streaming!)
tokenized_ds = tokenized_ds.with_format("torch")
print("Dataset tokenized. ✅\n")

# Use a standard DataLoader with the tokenized dataset
batch_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    num_workers=1,  # Already parallelized in map(), set to 0
    pin_memory=True,
    prefetch_factor=2
)

def load_checkpoint(path, model, optimizer, scaler):
    if os.path.exists(path):
        print(f"Restoring from {path}")
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
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

ckpt = torch.load(f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt")
model.load_state_dict(ckpt['model_state_dict'])
print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters. ✅\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model moved to {device} ✅:\n{torch.cuda.get_device_properties(device).name if device == 'cuda' else 'CPU'}\n")

optimizer = AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
    eps=1e-5
)

scaler = GradScaler()
loss_fn = torch.nn.CrossEntropyLoss()
start_step, min_loss = 0, float("inf")

for step, batch in enumerate(batch_loader, start=start_step):
    # x is all tokens except last, y is all tokens except first
    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)
    
    inputs = x[:, :-1]
    targets = y[:, 1:]

    with autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss = loss / ACCUMULATION_STEPS

    # Scaled backprop
    scaler.scale(loss).backward()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    curr_loss = loss.item() * ACCUMULATION_STEPS
    print(f"Step {step+1}, Loss: {curr_loss:.4f}, Tokens Processed: {((step+1)*BATCH_SIZE*CONTEXT_LENGTH):,}, lr: {LEARNING_RATE:.4e}")
    
    if (step + 1) % 1000 == 0:
        test_instruction = "Finish this sentence by saying what today's world is heavily influence by: Today's world is heavily"
        prompt = f"#####\nInstruction:\n{test_instruction}\n\n#####\nResponse:\n"
        
        print(f"Step {step+1}, generating from prompt:\n{prompt}")
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'min_loss': min(min_loss, curr_loss)
        }
        
        prompt_ids = tokenizer.encode(prompt).ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        generated_tokens = model.generate(
            x=prompt_tensor,
            max_new_tokens=140,
            temperature=0.2,
            top_k=50
        )
        
        decoded_output = tokenizer.decode(generated_tokens[0].tolist())
        print("\n" + "="*40)
        print("🤖 GENERATED TEXT:")
        print("="*40)
        print(decoded_output)
        print("="*40 + "\n")
        
        torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest_sft.pt")
        print(f"Latest checkpoint saved at step {step+1}")
        
        if curr_loss < min_loss:
            min_loss = curr_loss
            torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_best_sft.pt")
            print(f"Best checkpoint saved at step {step+1}, loss: {min_loss:.4f}")