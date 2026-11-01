import json
import pickle

import numpy as np
from gensim.models import Word2Vec

from custom_embedding import create_histogram_embedding


def load_custom_index(index_dir="custom"):
    # load custom embedding
    embeddings = np.load(
        f"{index_dir}/embeddings.npy"
    )

    with open(f"{index_dir}/chunks.json", "r") as f:
        chunks = json.load(f)

    with open(f"{index_dir}/sources.json", "r") as f:
        sources = json.load(f)

    word2vec_model = Word2Vec.load(
        f"{index_dir}/word2vec.model"
    )

    with open(f"{index_dir}/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    return embeddings, chunks, sources, word2vec_model, kmeans


def retrieve_chunks(
    question,
    embeddings,
    chunks,
    sources,
    word2vec_model,
    kmeans_model,
    k=4
):
    # use cosine similarity to retrieve chunks

    # generate query embedding
    query_embedding = create_histogram_embedding(
        question,
        word2vec_model,
        kmeans_model
    )

    if np.linalg.norm(query_embedding) == 0:
        print("No recognized words found in the question.")
        return []

    # calculate cosine similarity
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)

    similarities = np.zeros(len(embeddings))

    valid = embedding_norms > 0

    similarities[valid] = (
        embeddings[valid] @ query_embedding
    ) / (
        embedding_norms[valid] * query_norm
    )

    # get the top results
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "text": chunks[index],
            "page": sources[index]["page"],
            "similarity": float(similarities[index])
        })

    return results


def main():
    # load embeddings
    (
        embeddings,
        chunks,
        sources,
        word2vec_model,
        kmeans_model
    ) = load_custom_index()

    # prompt user for a question
    question = input("Enter a question: ")

    # retrieve chunks
    results = retrieve_chunks(
        question,
        embeddings,
        chunks,
        sources,
        word2vec_model,
        kmeans_model,
        k=4
    )

    # print results
    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(result["text"])


if __name__ == "__main__":
    main()