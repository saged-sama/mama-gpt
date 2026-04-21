from venv import logger
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers
import torch
from models.mamma import Mamma
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOCAB_SIZE = 30522
BATCH_SIZE = 1000
MAX_SAMPLES = 1_000_000
MODEL_NAME = "mama-gpt"
CONTEXT_LENGTH = 1024
d_model = 512
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

def token_stream(context_length):
    buffer = []

    for example in ds:
        tokens = tokenizer.encode(example["text"]).ids
        buffer.extend(tokens)

        while len(buffer) >= context_length:
            yield buffer[:context_length]
            buffer = buffer[context_length:]

def training_stream(context_length):
    for token_list in token_stream(context_length+1):
        input_ids = token_list[:-1]
        target_ids = token_list[1:]
        yield torch.tensor(input_ids), torch.tensor(target_ids)

def batch_loader(context_length, batch_size):
    batch_x, batch_y = [], []

    for x, y in training_stream(context_length):
        batch_x.append(x)
        batch_y.append(y)

        if len(batch_x) == batch_size:
            yield torch.stack(batch_x), torch.stack(batch_y)
            batch_x, batch_y = [], []

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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for step, (x, y) in enumerate(batch_loader(CONTEXT_LENGTH, BATCH_SIZE)):
    x, y = x.to(device), y.to(device)

    logits = model(x)

    B, T, V = logits.shape
    logits = logits.view(B*T, V)
    y = y.view(B*T)

    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info(f"Step {step}, Loss: {loss.item():.4f}, Tokens Processed: {(step+1)*BATCH_SIZE*CONTEXT_LENGTH}, Time: {torch.cuda.get_device_properties(device).name if device == 'cuda' else 'CPU'}")

    if step % 1000 == 0:
        logger.info(f"Saving checkpoint at step {step}...")
        torch.save(model.state_dict(), f"output/{MODEL_NAME}/checkpoint_{MODEL_NAME}_Latest.pt")
        logger.info("Checkpoint saved. ✅\n")

        model.eval()
        logger.info("Generating sample prediction...")
        with torch.no_grad():
            sample_input = torch.tensor(tokenizer.encode("Once upon a time").ids).unsqueeze(0).to(device)
            sample_output = model(sample_input)
            predicted_token_id = sample_output.argmax(dim=-1)[0, -1].item()
            predicted_token = tokenizer.decode([predicted_token_id])
            logger.info(f"Sample prediction: {predicted_token}")
        model.train()