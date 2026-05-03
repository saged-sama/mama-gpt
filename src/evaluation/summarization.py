from src.infer import load_model
from datasets import load_dataset
import torch
import json
import os

MAX_EXAMPLES = 100
CONTEXT_LENGTH = 1024
VOCAB_SIZE = 32_000

# Load model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model loaded on {device}\n")
print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")


# Load dataset
print("Loading dataset...")
ds = load_dataset(
    "EdinburghNLP/xsum",
    split="test",
    streaming=True
)
print("Dataset loaded. ✅\n")

# Prepare output directory
os.makedirs("logs", exist_ok=True)

# Store results
summaries_data = []
count = 0

print(f"Generating summaries for {MAX_EXAMPLES} examples...\n")

for example in ds:
    if count >= MAX_EXAMPLES:
        break
    
    try:
        article = example.get("document", "")
        
        if not article or len(article.strip()) < 10:
            continue
        
        # Create prompt for summarization
        prompt = f"<BOS>#####\nInstruction:\nSummarize this article: {article}\n\n#####\nResponse:\n"
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt).ids
        
        if max(prompt_ids) >= VOCAB_SIZE:
            print(f"Error: Token ID {max(prompt_ids)} exceeds vocab size {VOCAB_SIZE}")
            continue
        
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        # Generate summary
        with torch.no_grad():
            generated_tokens = model.generate(
                x=prompt_tensor,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50
            )
        
        # Decode summary
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        
        # Extract only the generated part (remove the prompt)
        if "Response:" in generated_text:
            summary = generated_text.split("Response:")[-1].strip()
        else:
            summary = generated_text[len(prompt):].strip()
        
        # Clean up the summary (remove EOS tokens if present)
        summary = summary.replace("<EOS>", "").strip()
        
        # Store result
        summaries_data.append({
            "article": article.strip(),
            "summary": summary
        })
        
        count += 1
        print(f"[{count}/{MAX_EXAMPLES}] Generated summary for article (length: {len(article)} chars)")
        
    except Exception as e:
        print(f"Error processing example: {e}")
        continue

# Save to JSON
output_path = "logs/summaries.json"
with open(output_path, "w") as f:
    json.dump(summaries_data, f, indent=4)

print(f"\n✅ Saved {count} summaries to {output_path}")