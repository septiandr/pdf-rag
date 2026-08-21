from pathlib import Path
import pickle

import faiss
import numpy as np


class VectorStore:

    def __init__(
        self,
        dimension: int,
        index_path: str = "storage/index.faiss",
        metadata_path: str = "storage/metadata.pkl",
    ):
        self.dimension = dimension

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index = faiss.IndexFlatIP(dimension)

        self.documents = []

    def add(self, embeddings, chunks):

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)

        for chunk in chunks:
            self.documents.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            })

    def save(self):

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(self.metadata_path, "wb") as file:
            pickle.dump(
                self.documents,
                file
            )

        print(f"Vector index saved: {self.index_path}")
        print(f"Metadata saved: {self.metadata_path}")

    def load(self):

        if not self.index_path.exists():
            return False

        if not self.metadata_path.exists():
            return False

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(self.metadata_path, "rb") as file:
            self.documents = pickle.load(file)

        return True

