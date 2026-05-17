from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

def rerank(query: str, docs, top_k: int = 5, threshold: float = 0.7) -> list[Document]:
    docs_curated = []
    for item in docs:
        doc, _ = item
        docs_curated.append(doc)
        
    pairs = [(query, doc.page_content) for doc in docs_curated]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(docs_curated, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:top_k]
                if score >= threshold
            ]