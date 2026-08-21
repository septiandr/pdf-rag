from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.search import Searcher

# Load index yang sudah ada
embedder = Embedder()
vector_store = VectorStore(dimension=1024)
loaded = vector_store.load()

if not loaded:
    print("Index belum ada, jalankan main.py dulu!")
    exit()

print(f"Loaded {vector_store.index.ntotal} vectors\n")

# Inisialisasi searcher
searcher = Searcher(vector_store, embedder)

# Test queries
test_queries = [
    "apa itu pancasila",
    "hak warga negara",
    "presiden",
    "pasal 1",
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: \"{query}\"")
    print(f"{'='*50}")

    results = searcher.search(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']:.4f}")
        print(f"   Cosine: {r['cosine_score']:.4f} | Keyword: {r['keyword_ratio']:.2f}")
        print(f"   {r['text'][:150]}...")
