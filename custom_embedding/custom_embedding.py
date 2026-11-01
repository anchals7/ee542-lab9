
import json
import pickle
import os
import re
import numpy as np

from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter


def normalize(text):
    # make all words same case
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def train_word2vec(chunks):
    # generate word2vec vectors
    normalized_chunks = [normalize(chunk) for chunk in chunks]

    model = Word2Vec(
        sentences=normalized_chunks,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=20,
        seed=42
    )

    return model


def train_kmeans(word2vec_model, k=5):
    # kmeans algorithm (k=5 here)
    words = list(word2vec_model.wv.index_to_key)

    if len(words) < k:
        raise ValueError("Not enough words to create K clusters.")

    word_vectors = np.array([
        word2vec_model.wv[word] for word in words
    ])

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(word_vectors)

    return kmeans


def create_histogram_embedding(text, word2vec_model, kmeans_model):
    # build normalized k-dimensional histogram vector

    # get chunks
    words = normalize(text)
    valid_words = [
        word for word in words
        if word in word2vec_model.wv
    ]

    # get clusters and build histogram
    k = kmeans_model.n_clusters
    histogram = np.zeros(k, dtype=np.float32)

    if not words or not valid_words:
        return histogram

    vectors = np.array([
        word2vec_model.wv[word] for word in valid_words
    ])

    labels = kmeans_model.predict(vectors)
    
    for label in labels:
        histogram[label] += 1

    # normalize histogram vector
    histogram /= len(words)

    return histogram


def main():
    # split data file into chunks and assign page numbers
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    sources = []

    reader = PdfReader("../data/licensing.pdf")

    print("splitting data\n")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            chunks.append(chunk)
            sources.append({
                "page": page_number
            })

    if not chunks:
        raise ValueError("No text could be extracted from the PDF.")

    print(f"split data into chunks\n")

    # use word2vec to generate vectors
    print("word2vec\n")
    word2vec = train_word2vec(chunks)

    # use k-means algorithm for clustering
    print(f"clustering\n")
    kmeans = train_kmeans(word2vec, k=5)

    # build histogram
    print("building histogram\n")
    embeddings = np.array([
        create_histogram_embedding(
            chunk,
            word2vec,
            kmeans
        )
        for chunk in chunks
    ])

    # save to folder
    print("saving to folder\n")
    output_folder = "custom"
    os.makedirs(output_folder, exist_ok=True)

    np.save(
        f"{output_folder}/embeddings.npy",
        embeddings
    )

    with open(f"{output_folder}/chunks.json", "w") as f:
        json.dump(chunks, f, indent=4)

    with open(f"{output_folder}/sources.json", "w") as f:
        json.dump(sources, f, indent=4)

    word2vec.save(
        f"{output_folder}/word2vec.model"
    )

    with open(f"{output_folder}/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)

if __name__ == "__main__":
    main()
