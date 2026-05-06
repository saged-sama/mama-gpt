import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class HybridSearch:
    def __init__(self, docs: list[Document], embeddings_model):
        self.docs = docs
        self.texts = [d.page_content for d in docs]

        # BM25 index
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Embed all documents
        self.embeddings_model = embeddings_model
        self.doc_embeddings = np.array(embeddings_model.embed_documents(self.texts))

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        # ---- BM25 ----
        tokenized_q = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_rank = np.argsort(bm25_scores)[::-1]

        # ---- Dense ----
        q_emb = np.array(self.embeddings_model.embed_query(query))
        dense_scores = cosine_similarity([q_emb], self.doc_embeddings)[0]
        dense_rank = np.argsort(dense_scores)[::-1]

        # ---- RRF ----
        return self.rrf([bm25_rank, dense_rank], top_k, k=60)

    def rrf(self, rankings, top_k, k=60) -> list[Document]:
        scores = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking[:top_k]):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (self.docs[i], score)
            for i, score in ranked[:top_k]
        ]