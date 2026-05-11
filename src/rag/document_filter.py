from datasets import load_dataset
from collections import Counter
from langchain_core.documents import Document

def get_top_docs(max_docs: int = 10):
    print("Loading Datasets...", end="")
    ds = load_dataset(
        "deepmind/narrativeqa",
        split="train"
    )
    print("✅")

    doc_counts = Counter()

    print(f"Retrieving top {max_docs} docs...", end="")
    for item in ds:
        doc_id = item["document"]["id"]
        doc_counts[doc_id] += 1
        
    top_doc_ids = set([
        doc_id for doc_id, _ in doc_counts.most_common(max_docs)
    ])
    
    queries_and_answers = {}
    seen_doc_ids = set()
    docs = []
    
    for item in ds:
        doc_id = item["document"]["id"]
        if doc_id in top_doc_ids:
            # Only create one Document per unique doc_id
            if doc_id not in seen_doc_ids:
                docs.append(Document(
                    page_content=item["document"]["summary"]["text"], 
                    metadata={
                        "id": doc_id,
                        "kind": item["document"]["kind"],
                        "url": item["document"]["url"]
                    }
                ))
                seen_doc_ids.add(doc_id)
            
            # Store all Q&A for this doc
            if queries_and_answers.get(doc_id) is None:
                queries_and_answers[doc_id] = []
            queries_and_answers[doc_id].append({
                "query": item["question"]["text"],
                "answer": item["answers"][0]["text"]
            })
    
    print("✅")
    import json
    print(json.dumps(queries_and_answers, indent=4))
    return docs, queries_and_answers