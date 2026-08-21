from utils.embedder import Embedder
from utils.cek_file import folder_has_content, get_pdf_files
from utils.pdf_loader import load_and_chunk_pdf_files
from utils.vector_store import VectorStore
from utils.search import Searcher
import numpy as np

# Initialize embedding model
embedder = Embedder()

# Inisialisasi VectorStore (dimensi Qwen3-Embedding-0.6B = 1024)
vector_store = VectorStore(dimension=1024)
vector_store.load()


if folder_has_content("./doc"):
    print("The folder has content.")
else:
    print("The folder is empty.")


pdf_files = get_pdf_files("./doc")


for pdf in pdf_files:
    print(f"Processing: {pdf}")

    chunks = load_and_chunk_pdf_files(pdf)

    print(f"Number of chunks: {len(chunks)}")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(texts)

    print(f"Chunks     : {len(chunks)}")
    print(f"Embeddings : {embeddings.shape}")

    # Tambahkan ke vector store
    vector_store.add(embeddings, chunks)

    print("Jumlah vector:", vector_store.index.ntotal)


# Simpan index ke disk
vector_store.save()


# === SEARCH ===
searcher = Searcher(vector_store, embedder)

print("\n--- Search Mode ---")
print("Ketik query untuk mencari, atau 'exit' untuk keluar.\n")

while True:
    query = input("Query: ")
    if query.lower() == "exit":
        break

    results = searcher.search(query, top_k=3)

    print(f"\nHasil pencarian untuk: \"{query}\"")
    print("-" * 50)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']:.4f}")
        print(f"   {r['text'][:200]}...")

    print()
