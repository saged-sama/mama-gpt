from datasets import load_dataset, concatenate_datasets
import hashlib

# Load the same datasets as in lora.py
print("Loading datasets...")

mbpp_ds = load_dataset("Muennighoff/mbpp", split="test")
small_code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
human_eval = load_dataset("openai/openai_humaneval", split="test")
caiss_mmlu = load_dataset("cais/mmlu", "college_computer_science", split="validation")

print("\n" + "="*70)
print("DATASET SIZES")
print("="*70)
print(f"MBPP (train):              {len(mbpp_ds)} samples")
print(f"CodeAlpaca (train):        {len(small_code_alpaca)} samples")
print(f"HumanEval (val):           {len(human_eval)} samples")
print(f"MMLU CS (val):             {len(caiss_mmlu)} samples")

# Check for overlaps between MBPP and HumanEval (most likely to overlap)
print("\n" + "="*70)
print("CHECKING FOR OVERLAPS: MBPP vs HumanEval")
print("="*70)

def get_text_hash(text):
    """Create a hash of text for comparison"""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

mbpp_hashes = {}
for idx, example in enumerate(mbpp_ds):
    text = example.get("text", "") + example.get("code", "")
    hash_val = get_text_hash(text)
    mbpp_hashes[hash_val] = idx

humaneval_hashes = {}
for idx, example in enumerate(human_eval):
    text = example.get("prompt", "") + example.get("canonical_solution", "")
    hash_val = get_text_hash(text)
    humaneval_hashes[hash_val] = idx

overlap_count = 0
overlapping_indices = []
for hash_val, he_idx in humaneval_hashes.items():
    if hash_val in mbpp_hashes:
        overlap_count += 1
        mbpp_idx = mbpp_hashes[hash_val]
        overlapping_indices.append((mbpp_idx, he_idx))

print(f"\nDirect text matches: {overlap_count} / {len(human_eval)} HumanEval samples")

if overlapping_indices:
    print("\nOverlapping samples found:")
    for mbpp_idx, he_idx in overlapping_indices[:5]:  # Show first 5
        print(f"  MBPP[{mbpp_idx}] ↔ HumanEval[{he_idx}]")

# Check dataset sources/metadata
print("\n" + "="*70)
print("DATASET METADATA & SOURCES")
print("="*70)
print(f"\n📊 MBPP:")
print(f"   - Source: Muennighoff/mbpp")
print(f"   - Split: test")
print(f"   - Type: Coding problems")

print(f"\n📊 CodeAlpaca:")
print(f"   - Source: sahil2801/CodeAlpaca-20k")
print(f"   - Split: train")
print(f"   - Type: Instruction-output pairs")

print(f"\n📊 HumanEval:")
print(f"   - Source: openai/openai_humaneval")
print(f"   - Split: test")
print(f"   - Type: Coding problems (OpenAI benchmark)")

print(f"\n📊 MMLU:")
print(f"   - Source: cais/mmlu college_computer_science")
print(f"   - Split: validation")
print(f"   - Type: Multiple choice questions")

# Risk assessment
print("\n" + "="*70)
print("OVERLAP RISK ASSESSMENT")
print("="*70)

print("""
⚠️  MODERATE RISK: MBPP (train) vs HumanEval (val)
   - Both are code generation benchmarks
   - Could share common programming problems
   - Problems may reference well-known algorithms (e.g., fibonacci, sorting)
   - Direct text overlap found: {}

✅ LOW RISK: CodeAlpaca (train) vs HumanEval (val)
   - CodeAlpaca is generic instruction dataset
   - HumanEval is specific coding benchmark
   - Unlikely to have significant overlap

✅ LOW RISK: MMLU (val)
   - Separate domain (multiple choice CS questions)
   - Specific to MMLU dataset
   - No significant overlap expected with code datasets

RECOMMENDATION:
- Consider using HumanEval test split for final testing
- Use a separate held-out test set not from these sources
- Consider using MBPP validation/train split for evaluation instead of test
""".format(overlap_count))
