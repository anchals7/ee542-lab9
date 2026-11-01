
import json
from pathlib import Path

import faiss
import numpy as np
import pymysql
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"

# Local embedding model
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# MySQL configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "chatuser",
    "password": "chatpass",
    "database": "groupchat",
    "charset": "utf8mb4",
}


def load_chunks():
    """Load HeartWise chunks and their source metadata from MySQL."""
    conn = pymysql.connect(**DB_CONFIG)

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT
                    c.id AS chunk_id,
                    c.text AS text,
                    c.page_number,
                    c.chunk_number,
                    d.filename,
                    d.title,
                    d.publisher,
                    d.topic,
                    d.source_url
                FROM hw_chunks c
                JOIN hw_documents d
                    ON c.document_id = d.id
                ORDER BY d.filename, c.chunk_number
            """)

            return cur.fetchall()
    finally:
        conn.close()


def build_index():
    print("Loading HeartWise chunks from MySQL...")
    chunks = load_chunks()

    if not chunks:
        raise RuntimeError("No HeartWise chunks found in the database.")

    print(f"Loaded {len(chunks)} chunks.")

    # Load the local embedding model.
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    # Generate normalized embeddings.
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Build a cosine-similarity index using inner product
    # because the embeddings are normalized.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save the index and corresponding metadata.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "heartwise.index"
    metadata_path = INDEX_DIR / "metadata.json"

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False, default=str)

    print("\nHeartWise vector index created successfully.")
    print(f"Chunks indexed: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    build_index()
