#Read every file in data/, split it into chunks, and save it to MySQL.

from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

import db

DATA_DIR = Path(__file__).parent / "data"

#set options according to manual
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
)


def read_pages(path):
    """Return a list with the text of each page of the file."""
    if path.suffix == ".pdf":
        return [page.extract_text() or "" for page in PdfReader(path).pages]

    if path.suffix in (".html", ".htm"):
        soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):  # drop code, keep readable text
            tag.decompose()
        return [soup.get_text("\n")]

    if path.suffix == ".txt":
        return [path.read_text(errors="ignore")]

    return []  # any other file type is skipped


def main():
    db.create_tables()
    conn = db.connect()
    cur = conn.cursor()

    for path in sorted(DATA_DIR.iterdir()):
        pages = read_pages(path)
        if not pages:
            continue

        # Skip files that were saved on an earlier run.
        cur.execute("SELECT id FROM documents WHERE filename = %s", (path.name,))
        if cur.fetchone():
            print(f"{path.name}: already saved, skipping")
            continue

        cur.execute(
            "INSERT INTO documents (filename, num_pages) VALUES (%s, %s)",
            (path.name, len(pages)),
        )
        document_id = cur.lastrowid  # the id MySQL just gave this document

        count = 0
        for page_number, text in enumerate(pages, start=1):
            for chunk in splitter.split_text(text):
                cur.execute(
                    "INSERT INTO chunks (document_id, page_number, text) VALUES (%s, %s, %s)",
                    (document_id, page_number, chunk),
                )
                count += 1

        print(f"{path.name}: {len(pages)} pages, {count} chunks")

    conn.close()


if __name__ == "__main__":
    main()
