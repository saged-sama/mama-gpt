import torch
from models.mamma import Mamma
from tokenizers import Tokenizer, decoders
import time

MODEL_NAME = "mama-gpt"
MODEL_PATH = f"output/{MODEL_NAME}/checkpoint_mama-gpt_123M_ds5B.pt"
TOKENIZER_PATH = f"output/{MODEL_NAME}/tokenizer.json"

VOCAB_SIZE = 50_000
D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 12
D_FF = 2048
CONTEXT_LENGTH = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.decoder = decoders.BPEDecoder()
    
    print(f"Initializing Mamma architecture...")
    model = Mamma(
        vocab_size=VOCAB_SIZE,
        dim=D_MODEL,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=D_FF
    )
    
    print(f"Loading weights from {MODEL_PATH}...")
    # Since you saved only the state_dict, we load it directly
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # If your checkpoint was the new 'dictionary' style, you'd use state_dict['model_state_dict']
    # But for your 5B token model, it's just the raw dict:
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    return model, tokenizer

def chat():
    model, tokenizer = load_model()
    print("\n" + "="*50)
    print("Mamma-GPT is ready. Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        prompt = input("User: ")
        if prompt.lower() in ['exit', 'quit', "q"]:
            break
        
        if not prompt.strip():
            continue

        # Encode input
        encoded = tokenizer.encode(prompt)
        x = torch.tensor([encoded.ids], dtype=torch.long).to(DEVICE)
        
        print("\nMamma: ", end="", flush=True)
        
        # Generation
        # Note: Ensure your generate() method in mamma.py uses use_cache=True now!
        start_time = time.time()
        
        with torch.no_grad():
            output_tensor = model.generate(
                x=x,
                max_new_tokens=128,
                temperature=0.7,
                top_k=50
            )
        
        # Decode and print
        full_text = tokenizer.decode(output_tensor[0].tolist())
        
        # Clean up output (optional: remove the prompt from the start)
        new_text = full_text[len(prompt):]
        print(new_text.strip())
        
        end_time = time.time()
        tokens_gen = output_tensor.shape[1] - x.shape[1]
        print(f"\n\n[Generated {tokens_gen} tokens in {end_time - start_time:.2f}s]")
        print("-" * 30)

if __name__ == "__main__":
    chat()