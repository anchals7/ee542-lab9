
"""Ingest HeartWise HTML sources into dedicated MySQL tables."""

import csv
from pathlib import Path

import pymysql
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Resolve paths relative to this script.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "heartwise"
METADATA_FILE = DATA_DIR / "sources.csv"

# Database configuration.
DB_CONFIG = {
    "host": "localhost",
    "user": "chatuser",
    "password": "chatpass",
    "database": "groupchat",
    "autocommit": True,
    "charset": "utf8mb4",
}

# Split article text into overlapping chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_metadata():
    """Load publisher, topic, and URL metadata."""
    with open(METADATA_FILE, newline="", encoding="utf-8") as file:
        return {
            row["filename"]: row
            for row in csv.DictReader(file)
        }


def extract_article(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        soup = BeautifulSoup(file, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem

    # Remove only elements that cannot contain useful article text.
    for element in soup.select(
        "script, style, noscript, svg, nav, footer, "
        "form, button"
    ):
        element.decompose()

    # Prefer main content, then article, then body.
    content = soup.find("main")

    if content is None:
        content = soup.find("article")

    if content is None:
        content = soup.body

    if content is None:
        content = soup

    text = content.get_text("\n", strip=True)

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ]

    return title, "\n".join(lines)


def create_tables(conn):
    """Create dedicated HeartWise tables."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hw_documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL UNIQUE,
                title VARCHAR(500),
                publisher VARCHAR(100),
                topic VARCHAR(255),
                source_url TEXT,
                num_pages INT NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hw_chunks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT NOT NULL,
                page_number INT NOT NULL DEFAULT 1,
                chunk_number INT NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES hw_documents(id)
                    ON DELETE CASCADE
            )
        """)


def ingest():
    """Read, clean, chunk, and save all HeartWise documents."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    metadata = load_metadata()
    conn = pymysql.connect(**DB_CONFIG)

    try:
        create_tables(conn)

        with conn.cursor() as cur:
            total_chunks = 0

            for path in sorted(DATA_DIR.glob("*.html")):
                if path.name not in metadata:
                    print(f"{path.name}: missing metadata, skipping")
                    continue

                title, text = extract_article(path)

                if len(text) < 300:
                    print(f"{path.name}: insufficient article text, skipping")
                    continue

                chunks = splitter.split_text(text)

                if not chunks:
                    print(f"{path.name}: no chunks generated, skipping")
                    continue

                source = metadata[path.name]

                # Replace any earlier ingestion of this document.
                cur.execute(
                    "DELETE FROM hw_documents WHERE filename = %s",
                    (path.name,)
                )

                cur.execute("""
                    INSERT INTO hw_documents
                    (filename, title, publisher, topic, source_url, num_pages)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    path.name,
                    title,
                    source["publisher"],
                    source["topic"],
                    source["source_url"],
                    1
                ))

                document_id = cur.lastrowid

                for chunk_number, chunk in enumerate(chunks, start=1):
                    cur.execute("""
                        INSERT INTO hw_chunks
                        (document_id, page_number, chunk_number, text)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        document_id,
                        1,
                        chunk_number,
                        chunk
                    ))

                total_chunks += len(chunks)

                print(
                    f"{path.name}: "
                    f"{len(text):,} characters, "
                    f"{len(chunks)} chunks"
                )

            print(f"\nIngestion complete: {total_chunks} chunks created.")

    finally:
        conn.close()


if __name__ == "__main__":
    ingest()

