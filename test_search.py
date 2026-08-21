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
    print(f"\n{'='*60}")
    print(f"🔍 Query: \"{query}\"")
    print(f"{'='*60}")

    results = searcher.search(query, top_k=3)

    for i, r in enumerate(results, 1):
        # Bersihkan teks
        text = r["text"].replace("\n", " ").strip()
        text = " ".join(text.split())

        # Ambil potongan yang relevan
        if len(text) > 300:
            text = text[:300] + "..."

        # Deteksi pasal
        pasal = ""
        for line in r["text"].split("\n"):
            if "Pasal" in line:
                pasal = line.strip()
                break

        print(f"\n{'─' * 60}")
        print(f"📄 Hasil #{i}")
        if pasal:
            print(f"📌 {pasal}")
        print(f"   {text}")
        print(f"   ")
        print(f"   Skor relevansi: {r['score']:.0%}")
