
import json
import pickle
import os
import re
import numpy as np

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.cluster import KMeans

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter


# normalize text
def normalize(text):
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


# train word2vec
def train_word2vec(chunks, use_phrases=True, sg=1):
    normalized_chunks = [normalize(chunk) for chunk in chunks]

    phraser = None

    # detect common phrases
    if use_phrases:
        phrases = Phrases(
            normalized_chunks,
            min_count=2,
            threshold=5
        )

        phraser = Phraser(phrases)

        normalized_chunks = [
            phraser[words] for words in normalized_chunks
        ]

    model = Word2Vec(
        sentences=normalized_chunks,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=sg,
        sample=0.001,
        negative=5,
        epochs=20,
        seed=42
    )

    return model, phraser


# train k-means
def train_kmeans(word2vec_model, k=5):
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


# create normalized histogram embedding
def create_histogram_embedding(
    text, word2vec_model, kmeans_model, phraser=None
):
    # get original words
    words = normalize(text)
    total_words = len(words)

    # apply phrase detection
    if phraser is not None:
        words = phraser[words]

    valid_words = [
        word for word in words
        if word in word2vec_model.wv
    ]

    # build histogram
    k = kmeans_model.n_clusters
    histogram = np.zeros(k, dtype=np.float32)

    if total_words == 0 or not valid_words:
        return histogram

    vectors = np.array([
        word2vec_model.wv[word] for word in valid_words
    ])

    labels = kmeans_model.predict(vectors)

    for label in labels:
        histogram[label] += 1

    # normalize by original word count
    histogram /= total_words

    return histogram


# calculate cosine similarity
def cosine_similarity(vector1, vector2):
    denominator = (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(vector1, vector2) / denominator)


# retrieve the most relevant chunks
def retrieve_chunks(
    question,
    embeddings,
    chunks,
    sources,
    word2vec_model,
    kmeans_model,
    phraser,
    top_k=4
):
    query_embedding = create_histogram_embedding(
        question,
        word2vec_model,
        kmeans_model,
        phraser
    )

    similarities = [
        cosine_similarity(query_embedding, embedding)
        for embedding in embeddings
    ]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "page": sources[index]["page"],
            "similarity": similarities[index],
            "text": chunks[index]
        })

    return results


# run one experiment
def run_experiment(
    chunks,
    sources,
    word2vec_model,
    phraser,
    experiment_name,
    k,
    output_root
):
    print(f"\nTraining K-means with K={k}")

    kmeans = train_kmeans(word2vec_model, k=k)

    # generate histogram embeddings
    embeddings = np.array([
        create_histogram_embedding(
            chunk,
            word2vec_model,
            kmeans,
            phraser
        )
        for chunk in chunks
    ])

    # create output folder
    output_folder = os.path.join(
        output_root,
        f"{experiment_name}_k{k}"
    )

    os.makedirs(output_folder, exist_ok=True)

    # save embeddings and data
    np.save(
        os.path.join(output_folder, "embeddings.npy"),
        embeddings
    )

    with open(os.path.join(output_folder, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=4)

    with open(os.path.join(output_folder, "sources.json"), "w") as f:
        json.dump(sources, f, indent=4)

    # save models
    word2vec_model.save(
        os.path.join(output_folder, "word2vec.model")
    )

    with open(os.path.join(output_folder, "kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)

    if phraser is not None:
        phraser.save(
            os.path.join(output_folder, "phraser.model")
        )

    return (
        embeddings,
        kmeans,
        output_folder
    )


def main():
    # split PDF into chunks and assign page numbers
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    sources = []

    reader = PdfReader("../data/licensing.pdf")

    print("Splitting data\n")

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

    print(f"Split data into {len(chunks)} chunks\n")

    # experiment configurations
    experiments = [
        {
            "name": "baseline",
            "use_phrases": False,
            "sg": 1
        },
        {
            "name": "phrases_skipgram",
            "use_phrases": True,
            "sg": 1
        },
        {
            "name": "phrases_cbow",
            "use_phrases": True,
            "sg": 0
        }
    ]

    # test different numbers of clusters
    k_values = [5, 10, 25, 50]

    # same retrieval question for every configuration
    question = "What rules must be followed for Keysight EEsof’s new releases to start up?"

    output_root = "custom_comparison"
    os.makedirs(output_root, exist_ok=True)

    comparison_results = []

    # run each experiment
    for experiment in experiments:
        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {name}")
        print("=" * 60)

        # train Word2Vec 
        print("\nTraining Word2Vec")

        word2vec, phraser = train_word2vec(
            chunks,
            use_phrases=experiment["use_phrases"],
            sg=experiment["sg"]
        )

        # test each K value
        for k in k_values:
            embeddings, kmeans, output_folder = run_experiment(
                chunks,
                sources,
                word2vec,
                phraser,
                name,
                k,
                output_root
            )

            # retrieve chunks
            results = retrieve_chunks(
                question,
                embeddings,
                chunks,
                sources,
                word2vec,
                kmeans,
                phraser,
                top_k=4
            )

            print(f"\n{name.upper()} | K={k}")
            print("-" * 40)

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. Page {result['page']} | "
                    f"Similarity: {result['similarity']:.4f}"
                )

                print(result["text"][:200].replace("\n", " "))
                print()

            # record best result for comparison
            comparison_results.append({
                "experiment": name,
                "K": k,
                "top_page": results[0]["page"],
                "top_similarity": results[0]["similarity"],
                "top_text": results[0]["text"][:300],
                "output_folder": output_folder
            })

    # print comparison summary
    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Experiment':<22}"
        f"{'K':<8}"
        f"{'Top Page':<12}"
        f"{'Similarity':<12}"
    )

    print("-" * 75)

    for result in comparison_results:
        print(
            f"{result['experiment']:<22}"
            f"{result['K']:<8}"
            f"{result['top_page']:<12}"
            f"{result['top_similarity']:.4f}"
        )

    # save comparison results
    with open(
        os.path.join(output_root, "comparison_results.json"),
        "w"
    ) as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\nComparison results saved in {output_root}/")


if __name__ == "__main__":
    main()
