from datasets import load_dataset
from collections import Counter

print("Loading Datasets...", end="")
ds = load_dataset(
    "deepmind/narrativeqa",
    split="train"
)
print("✅")

doc_counts = Counter()

print("Retrieving top 10 docs...", end="")
for item in ds:
    doc_id = item["document"]["id"]
    doc_counts[doc_id] += 1
    
top_10_doc_ids = set([
    doc_id for doc_id, _ in doc_counts.most_common(10)
])

top_docs = ds.filter(lambda x: x["document"]["id"] in top_10_doc_ids)
print("✅")