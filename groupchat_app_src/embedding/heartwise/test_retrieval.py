
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 4

QUESTIONS = [
    "What are the major risk factors for heart disease?",
    "How can someone prevent high blood pressure?",
    "What foods are recommended for a heart-healthy diet?",
    "What happens when blood flows through the heart?",
    "How much physical activity should adults get?",
]


def main():
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_DIR / "heartwise.index"))

    with open(INDEX_DIR / "metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype("float32")

        scores, indices = index.search(query_embedding, TOP_K)

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0:
                continue

            chunk = metadata[idx]

            print(f"\nResult {rank} | Similarity: {score:.4f}")
            print("Topic:", chunk["topic"])
            print("Publisher:", chunk["publisher"])
            print("Source:", chunk["filename"])
            print("Text:", chunk["text"][:450].replace("\n", " "))


if __name__ == "__main__":
    main()
