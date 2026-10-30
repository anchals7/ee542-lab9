"""Embed every chunk in MySQL with OpenAI and save a FAISS vector store."""
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import db

load_dotenv()  # loads OPENAI_API_KEY from .env


def get_vectorstore(texts, sources):
    """Embed the chunks with OpenAI and store them in FAISS."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(texts, embeddings, metadatas=sources)


def build_index():
    """Embed every chunk in MySQL and save the result to faiss_index/."""
    # Get every chunk's text, plus the filename and page it came from.
    cur = db.connect().cursor()
    cur.execute("""
        SELECT chunks.text, documents.filename, chunks.page_number
        FROM chunks JOIN documents ON chunks.document_id = documents.id
    """)
    rows = cur.fetchall()

    texts = [text for text, filename, page in rows]
    sources = [{"source": filename, "page": page} for text, filename, page in rows]

    vectorstore = get_vectorstore(texts, sources)
    vectorstore.save_local("faiss_index")
    return vectorstore


if __name__ == "__main__":
    vectorstore = build_index()
    print(f"Embedded {vectorstore.index.ntotal} chunks and saved them to faiss_index/")