from sentence_transformers import SentenceTransformer
from typing import List


class LocalEmbeddings:
    """
    Free, local embeddings using SentenceTransformers.
    No API keys, no rate limits, no caching - just fast local inference.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
                - all-MiniLM-L6-v2: Fast, small, good quality (22MB)
                - all-mpnet-base-v2: Slower but better quality (438MB)
        """
        print(f"[EMBED] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"[EMBED] ✓ Model loaded")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode(text, convert_to_tensor=False).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents at once (faster)."""
        print(f"[EMBED] Embedding {len(texts)} documents...")
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

