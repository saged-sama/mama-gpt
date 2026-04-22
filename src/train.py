from venv import logger
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers
import torch
from models.mamma import Mamma
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOCAB_SIZE = 50_000
BATCH_SIZE = 1000
MAX_SAMPLES = 1_00_000
MODEL_NAME = "mama-gpt"
CONTEXT_LENGTH = 1024
NUM_WORKERS = 4
LR_SCHEDULER_STEP_SIZE = 30
MAX_NORM = 1.0
ACCUMULATION_STEPS=4
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.1
d_model = 768
num_heads = 8
num_layers = 6
d_ff = 2048

logging.log(level=logging.INFO, msg=f"Training {MODEL_NAME} with\nvocab size = {VOCAB_SIZE}\nbatch size = {BATCH_SIZE}\nmax samples = {MAX_SAMPLES}\ncontext length = {CONTEXT_LENGTH}\nmodel dimension = {d_model}\nnumber of heads = {num_heads}\nnumber of layers = {num_layers}\nfeedforward dimension = {d_ff}\n\n")

logging.log(level=logging.INFO, msg="Loading dataset...")
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)
logging.log(level=logging.INFO, msg="Dataset loaded. ✅")

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

logging.log(level=logging.INFO, msg="Training tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
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

tokenizer.save(f"output/{MODEL_NAME}/tokenizer.json")
logging.log(level=logging.INFO, msg="Tokenizer trained and saved. ✅")
logging.log(level=logging.INFO, msg=f"Total Vocab Size: {tokenizer.get_vocab_size()}\n\n")


def tokenize_fn(examples):
    """Tokenize examples in parallel across num_workers."""
    return tokenizer(examples["text"], 
                     truncation=True, 
                     max_length=CONTEXT_LENGTH + 1, 
                     padding="max_length")

# Map the dataset (this stays streaming!)
logging.log(level=logging.INFO, msg="Tokenizing dataset with parallel workers...")
tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
tokenized_ds = tokenized_ds.with_format("torch")
logging.log(level=logging.INFO, msg="Dataset tokenized. ✅\n")

# Use a standard DataLoader with the tokenized dataset
batch_loader = DataLoader(
    tokenized_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,  # Already parallelized in map(), set to 0
    pin_memory=True,
    prefetch_factor=2
)

model = Mamma(
    vocab_size=VOCAB_SIZE,
    dim=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=d_ff
)
logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters. ✅\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logger.info(f"Model moved to {device}. ✅\n")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_sched = lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=0.1)
scaler = GradScaler()
loss_fn = torch.nn.CrossEntropyLoss()

for step, batch in enumerate(batch_loader):
    # x is all tokens except last, y is all tokens except first
    x = batch["input_ids"][:, :-1].to(device)
    y = batch["input_ids"][:, 1:].to(device)

    with autocast(device_type=device, dtype=torch.float16):
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

    logger.info(f"Step {step+1}, Loss: {loss.item() * ACCUMULATION_STEPS:.4f}, Tokens Processed: {(step+1)*BATCH_SIZE*CONTEXT_LENGTH}, Device: {torch.cuda.get_device_properties(device).name if device == 'cuda' else 'CPU'}")
    
    if (step + 1) % 1000 == 0:
        prompt = "Today's world is heavily "
        
        logger.info(f"Step {step+1}, generating from prompt: {prompt}")
        
        prompt_tokens = tokenizer.encode(prompt)
        
        generated_tokens = model.generate(
            x=prompt_tokens,
            max_new_tokens=40,
            temperature=1,
            top_k=None
        )
        
        generated_text = tokenizer.decode(generated_text)
        logger.info(f"Generated Text: \n{generated_text}")
        