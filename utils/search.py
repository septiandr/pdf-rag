import numpy as np
from utils.vector_store import VectorStore
from utils.embedder import Embedder
from utils.query_enhancer import QueryEnhancer


class Searcher:

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_enhancer = QueryEnhancer()

    def search(self, query: str, top_k: int = 5, use_enhancement: bool = True):
        # Query Enhancement
        if use_enhancement:
            enhanced_query = self.query_enhancer.enhance(query)
            print(f"Enhanced query: \"{enhanced_query}\"")
        else:
            enhanced_query = query

        query_embedding = self.embedder.embed_query(enhanced_query)

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        # Ambil lebih banyak kandidat untuk re-ranking
        candidate_k = min(top_k * 3, self.vector_store.index.ntotal)
        scores, indices = self.vector_store.index.search(
            query_vector,
            candidate_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            document = self.vector_store.documents[index]

            # Re-ranking: kombinasi cosine similarity + keyword match
            text_lower = document["text"].lower()
            query_words = query.lower().split()

            # Hitung keyword match ratio
            keyword_hits = sum(1 for word in query_words if word in text_lower)
            keyword_ratio = keyword_hits / len(query_words) if query_words else 0

            # Skor akhir = 70% similarity + 30% keyword match
            final_score = (float(score) * 0.7) + (keyword_ratio * 0.3)

            results.append({
                "score": final_score,
                "cosine_score": float(score),
                "keyword_ratio": keyword_ratio,
                "text": document["text"],
                "metadata": document["metadata"],
            })

        # Sort berdasarkan skor akhir
        results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplication berdasarkan text
        seen_texts = set()
        unique_results = []

        for r in results:
            # Gunakan 100 karakter pertama sebagai key
            text_key = r["text"][:100].strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(r)

        return unique_results[:top_k]
