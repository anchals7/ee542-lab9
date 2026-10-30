# documents: one row per file
# chunks:    one row per ~500-character piece of a file
# Run once to create the tables:  python db.py

import pymysql


def connect():
    return pymysql.connect(
        host="localhost",
        user="chatuser",
        password="chatpass",
        database="groupchat",
        autocommit=True,  # save every change immediately
    )


def create_tables():
    db = connect()
    cur = db.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id        INT AUTO_INCREMENT PRIMARY KEY,
            filename  VARCHAR(255) NOT NULL UNIQUE,
            num_pages INT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            page_number INT NOT NULL,
            text        TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)

    db.close()


if __name__ == "__main__":
    create_tables()
    print("Tables ready: documents, chunks")
