# PDF RAG (Retrieval-Augmented Generation)

Sistem pencarian dokumen PDF menggunakan vector embedding dan FAISS untuk pencarian semantik.

## Fitur

- **Query Enhancement** — Otomatis memperkaya query dengan konteks hukum Indonesia
- **Re-ranking** — Menggabungkan cosine similarity dan keyword match untuk hasil lebih akurat
- **Semantic Chunking** — Split teks berdasarkan section (BAB, Pasal, dll)
- **Vector Store** — Index tersimpan di disk dan bisa di-load ulang
- **Deduplication** — Menghapus hasil search yang duplikat

## Instalasi

```bash
# Clone repository
git clone <url>
cd pdf-rag

# Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Struktur Project

```
pdf-rag/
├── main.py                 # Entry point (indexing + search)
├── test_search.py          # Script test search
├── requirements.txt        # Dependencies
├── doc/                    # Folder untuk file PDF
│   └── UUD45.pdf
├── storage/                # Penyimpanan index
│   ├── index.faiss
│   └── metadata.pkl
└── utils/
    ├── cek_file.py         # Helper cek folder & file
    ├── embedder.py         # Embedding model (Qwen3-Embedding-0.6B)
    ├── pdf_loader.py       # Load & chunk PDF
    ├── query_enhancer.py   # Query enhancement
    ├── search.py           # Search dengan re-ranking
    └── vector_store.py     # FAISS vector store
```

## Cara Penggunaan

### 1. Indexing PDF

```bash
python3 main.py
```

Program akan:
- Load semua PDF dari folder `doc/`
- Split teks menjadi chunks
- Generate embeddings
- Simpan index ke folder `storage/`

### 2. Search (Interactive)

Setelah indexing selesai, program akan masuk search mode:

```
--- Search Mode ---
Ketik query untuk mencari, atau 'exit' untuk keluar.

Query: apa itu pancasila

Hasil pencarian untuk: "apa itu pancasila"
--------------------------------------------------

1. Score: 0.6295
   PENJELASAN TENTANG UNDANG-UNDANG DASAR...

Query: exit
```

### 3. Search (Script)

```python
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.search import Searcher

embedder = Embedder()
vector_store = VectorStore(dimension=1024)
vector_store.load()

searcher = Searcher(vector_store, embedder)

results = searcher.search("hak warga negara", top_k=3)

for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Text: {r['text'][:200]}...")
```

### 4. Test Search

```bash
python3 test_search.py
```

## Komponen Utama

### Embedder
Menggunakan model **Qwen3-Embedding-0.6B** dari HuggingFace untuk generate vector embeddings.

### VectorStore
Menyimpan dan mengelola index FAISS. Support:
- `load()` — Load index dari disk
- `add()` — Tambah embedding baru
- `save()` — Simpan index ke disk

### Searcher
Kelas utama untuk pencarian:
- Query Enhancement — Memperkaya query sebelum di-embed
- Re-ranking — Skor gabungan (70% cosine + 30% keyword)
- Deduplication — Menghapus hasil duplikat

### QueryEnhancer
Menambah konteks otomatis berdasarkan keyword:
- `uud` → `Undang-Undang Dasar`
- `presiden` → `Presiden Republik Indonesia`
- `hak` → `hak asasi manusia`
- dst.

## Dependencies

- `sentence-transformers` — Embedding model
- `faiss-cpu` — Vector similarity search
- `langchain` — Document loading & text splitting
- `torch` — Deep learning framework

## License

MIT
