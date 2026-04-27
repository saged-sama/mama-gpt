import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.api.model import LM
from src.models.mamma import Mamma
import torch
from transformers import PreTrainedTokenizerFast
import json

class CustomLM(LM):
    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size

    def loglikelihood(self, requests):
        # Return list of (logprob, is_greedy) tuples
        results = []
        for req in requests:
            if hasattr(req, "arguments"):
                context, continuation = req.arguments
            elif hasattr(req, "args"):
                context, continuation = req.args
            else:
                context, continuation = req

            context = context or ""
            continuation = continuation or ""

            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            continuation_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

            if len(continuation_ids) == 0:
                results.append((0.0, True))
                continue

            if len(context_ids) == 0:
                if self.tokenizer.bos_token_id is not None:
                    context_ids = [self.tokenizer.bos_token_id]
                elif self.tokenizer.eos_token_id is not None:
                    context_ids = [self.tokenizer.eos_token_id]

            input_ids = torch.tensor([context_ids + continuation_ids], device=self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # logits at position t predict token t+1
            logprobs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            start = len(context_ids) - 1
            end = start + len(continuation_ids)

            continuation_logprobs = logprobs[0, start:end, :]
            target = torch.tensor(continuation_ids, device=self.model.device)

            token_logprobs = continuation_logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
            total_logprob = token_logprobs.sum().item()

            greedy_tokens = continuation_logprobs.argmax(dim=-1)
            is_greedy = bool(torch.equal(greedy_tokens, target))

            results.append((total_logprob, is_greedy))

        return results

    def generate_until(self, requests):
        # Return list of generated strings
        results = []
        for context, until in requests:
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(input_ids, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            results.append(generated_text)
        return results

    def loglikelihood_rolling(self, requests):
        # Return list of (logprob, is_greedy) tuples
        results = []
        for request in requests:
            input_ids = self.tokenizer.encode(request, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            logprob = torch.nn.functional.log_softmax(logits, dim=-1)
            total_logprob = logprob[0, -1, input_ids[0, -1]].item()
            results.append((total_logprob, False))
        return results

    @property
    def batch_size(self):
        return self._batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Mamma(
    vocab_size=50000,
    dim=768,
    context_length=1024,
    num_layers=12,
    num_heads=12,
    hidden_dim=3072
)

checkpoint = torch.load("output/mama-gpt/checkpoint_mama-gpt_latest.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device=device)
model.eval()

tokenizer = PreTrainedTokenizerFast(tokenizer_file="output/mama-gpt/tokenizer.json")

model.device = torch.device(device)
model.name_or_path = "custom"
model.tie_weights = lambda: None

model_obj = CustomLM(model=model, tokenizer=tokenizer, batch_size=16)
# if model_obj.prefix_token_id is None:
#     model_obj.prefix_token_id = []

models = [
    {
        "name": "mama-gpt",
        "lm_object": model_obj,
    },
    {
        "name": "gpt2",
        "lm_object": HFLM(pretrained="gpt2", device=device)
    },
    {
        "name": "phi-1.5", 
        "lm_object": HFLM(pretrained="microsoft/phi-1_5", device=device)
    }
]

results_summary = {}

specific_tasks = [
    "mmlu_college_computer_science", 
    "mmlu_college_mathematics", 
    "mmlu_jurisprudence"
]

print_string = ""

for model_info in models:
    print(f"--- Evaluating {model_info['name']} ---")
    
    model_obj = model_info['lm_object']

    results = lm_eval.simple_evaluate(
        model=model_obj,
        tasks=specific_tasks,
        num_fewshot=5,
        device="cuda:0",
        batch_size="auto"
    )
    
    scores = results['results']
    
    print_string = print_string + f"Evaluation for model: {model_info['name']}\n"
    
    for task in specific_tasks:
        alias = scores[task]["alias"]
        score = scores[task]["acc,none"] * 100
        print_string = print_string + (f"Task: mmlu, Alias: {alias}, Score: {score:.2f}%\n")
        
        if results_summary.get(model_info['name']) is None:
            results_summary[model_info['name']] = []
        results_summary[model_info['name']].append({
            "alias": alias,
            "score": score
        })
        
    print_string = print_string + "\n"
    
print(print_string)

with open("logs/mmlu_eval.json", mode="w", encoding="UTF-8") as f:
    f.write(json.dumps(results_summary, indent=4))