import os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from src.models.mamma import Mamma

# --- Configuration ---
VOCAB_SIZE = 50_000
BATCH_SIZE = 16
MAX_SAMPLES = 50_000
MODEL_NAME = "mama-gpt"
MODEL_PATH = f"output/{MODEL_NAME}"
CONTEXT_LENGTH = 1024
MAX_NORM = 1.0
ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 3072

print(f"Training {MODEL_NAME} with\nvocab size = {VOCAB_SIZE}\nbatch size = {BATCH_SIZE}\nmax samples = {MAX_SAMPLES}\ncontext length = {CONTEXT_LENGTH}\n\n")

# --- Dataset Loading ---
print("Loading dataset...")
ds = load_dataset(
    "glnmario/news-qa-summarization",
    split="train",
    streaming=True
)
print("Dataset loaded. ✅")

# --- Prompt Formatting ---
def format_prompt(example):
    system_prompt = example.get("system", "")
    story = example.get("story", "")
    summary = example.get("summary", "")
    
    prompt = ""
    if system_prompt and system_prompt.strip():
        prompt = f"#####\nSystem:\n{system_prompt}\n\n"
        
    # Injecting BOS and EOS directly into the text template
    prompt += f"<BOS>#####\nInstruction:\n Summarize this article: {story}\n\n#####\nResponse:\n{summary}<EOS>"
    return prompt

# --- Tokenizer Training ---
def batch_iterator(batch_size=1000, max_samples=10_000):
    batch = []
    count = 0
    for example in ds:
        # Train on formatted prompts so it learns the template tokens efficiently
        batch.append(format_prompt(example))
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
        batch_iterator(batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES),
        trainer=trainer
    )
    tokenizer.save(f"{MODEL_PATH}/tokenizer.json")
    print("Tokenizer trained and saved. ✅")

PAD_ID = tokenizer.token_to_id("<PAD>")

# --- Tokenization & Masking Logic ---
def tokenize_sft_fn(example):
    full_text = format_prompt(example)
    encodings = tokenizer.encode(full_text)
    ids = encodings.ids
    
    if len(ids) > CONTEXT_LENGTH:
        ids = ids[:CONTEXT_LENGTH]
        
    # No -100 masking for now, train on the whole sequence
    labels = list(ids)
    
    padding_len = CONTEXT_LENGTH - len(ids)
    
    # input_ids get PAD_ID, labels get -100 ONLY for padding
    input_ids = ids + [PAD_ID] * padding_len
    labels = labels + [-100] * padding_len
    
    # Return standard python lists, letting the datasets library handle tensor conversion
    return {
        "input_ids": input_ids,
        "labels": labels
    }

tokenized_ds = ds.map(tokenize_sft_fn, remove_columns=ds.column_names)
tokenized_ds = tokenized_ds.with_format("torch")

# num_workers=0 is standard for streaming datasets on a single node to prevent data duplication
batch_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    num_workers=0,  
    pin_memory=True,
    prefetch_factor=None 
)

# --- Checkpoint Loading Helper ---
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if os.path.exists(path):
        print(f"Restoring from {path}...")
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        return ckpt.get('step', 0), ckpt.get('min_loss', float('inf'))
    return 0, float('inf')

# --- Model Initialization ---
model = Mamma(
    vocab_size=VOCAB_SIZE,
    dim=d_model,
    context_length=CONTEXT_LENGTH,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=d_ff
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
    eps=1e-5
)

# Added Cosine Annealing Scheduler (Assuming max_samples approx steps)
total_estimated_steps = (MAX_SAMPLES // (BATCH_SIZE * ACCUMULATION_STEPS)) + 1
scheduler = CosineAnnealingLR(optimizer, T_max=total_estimated_steps, eta_min=1e-6)

loss_fn = torch.nn.CrossEntropyLoss()

# Load latest state if available
latest_ckpt_path = f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest.pt"
start_step, min_loss = load_checkpoint(latest_ckpt_path, model, optimizer, scheduler)

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters on {device}. ✅\n")

# --- Training Loop ---
model.train()
running_loss = 0.0

for step, batch in enumerate(batch_loader, start=start_step):
    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)
    
    inputs = x[:, :-1]
    targets = y[:, 1:]

    # Use bfloat16 for newer GPUs, float16 if you encounter issues on older hardware
    with autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        scaled_loss = loss / ACCUMULATION_STEPS

    # Standard backward pass (no scaler for bfloat16)
    scaled_loss.backward()
    running_loss += scaled_loss.item()

    if (step + 1) % ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step+1}, Loss: {running_loss:.4f}, Tokens Processed: {((step+1)*BATCH_SIZE*CONTEXT_LENGTH):,}, lr: {current_lr:.4e}")
        
        # Track for saving best model later
        curr_eval_loss = running_loss
        running_loss = 0.0
    
    # --- Evaluation / Generation Checkpoint ---
    if (step + 1) % 1000 == 0:
        model.eval()
        test_instruction = "Finish this sentence by saying what today's world is heavily influence by: Today's world is heavily"
        prompt = f"<BOS>#####\nInstruction:\n{test_instruction}\n\n#####\nResponse:\n"
        
        print(f"\nStep {step+1}, generating from prompt:\n{prompt}")
        
        prompt_ids = tokenizer.encode(prompt).ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        # Ensure generation doesn't update gradients
        with torch.no_grad():
            generated_tokens = model.generate(
                x=prompt_tensor,
                max_new_tokens=140,
                temperature=0.2,
                top_k=50,
                pad_token_id="<EOS>"
            )
        
        decoded_output = tokenizer.decode(generated_tokens[0].tolist())
        print("="*40 + "\n🤖 GENERATED TEXT:\n" + "="*40)
        print(decoded_output)
        print("="*40 + "\n")
        
        checkpoint = {
            'step': step + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min(min_loss, curr_eval_loss)
        }
        
        torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_latest_sft.pt")
        print(f"Latest checkpoint saved at step {step+1}")
        
        if curr_eval_loss < min_loss:
            min_loss = curr_eval_loss
            torch.save(checkpoint, f"{MODEL_PATH}/checkpoint_{MODEL_NAME}_best_sft.pt")
            print(f"🌟 Best checkpoint saved at step {step+1}, loss: {min_loss:.4f}")
            
        model.train()