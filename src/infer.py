import torch
from src.models.mamma import Mamma
from tokenizers import Tokenizer, decoders
import time

MODEL_NAME = "mama-gpt-larger"
MODEL_PATH = f"output/{MODEL_NAME}/checkpoint_{MODEL_NAME}_latest_sft.pt"
TOKENIZER_PATH = f"output/{MODEL_NAME}/tokenizer.json"

VOCAB_SIZE = 32_000
D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 12
D_FF = int(8 * 768 / 3)
CONTEXT_LENGTH = 3072

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.decoder = decoders.ByteLevel()
    
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
    # model.load_state_dict(state_dict['model_state_dict'])
    model.load_state_dict(state_dict)
    
    
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    
    print(f"Model loaded with parameters count: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer

def format_prompt(context):
    return (
        f"<context>{context}</context>\n"
        "<summary>"
    )

def chat():
    model, tokenizer = load_model()
    print("\n" + "="*50)
    print("Mamma-GPT is ready. Type 'exit' to quit.")
    print("End your input with '#$%@'.")
    print("="*50 + "\n")

    while True:
        print("Story: ")
        context = input()
        # print("Question: ")
        # question = input()

        prompt = format_prompt(context=context)

        encoded = tokenizer.encode(prompt)
        x = torch.tensor([encoded.ids], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output_tensor = model.generate(
                x=x,
                max_new_tokens=512,   # answers are short, no need for 512
                temperature=0.1,      # lower = more focused/factual
                top_k=10,
                eos_token_id=tokenizer.token_to_id("<EOS>")
            )

        full_text = tokenizer.decode(output_tensor[0].tolist())
        # strip everything up to and including "Response:\n"
        answer = full_text.split("Response:")[-1].strip()
        print(f"\nMamma: {answer}")

if __name__ == "__main__":
    chat()