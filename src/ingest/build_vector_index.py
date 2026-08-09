import json
import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

RAW_PAPERS_PATH = "data/raw_papers.json"
POSTGRES_URL = os.getenv("POSTGRES_URL")

_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_connection():
    conn = psycopg.connect(POSTGRES_URL, autocommit=True)
    register_vector(conn)
    return conn


def create_table(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_chunks (
            chunk_id TEXT PRIMARY KEY,
            arxiv_id TEXT NOT NULL,
            title TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            published DATE,
            embedding vector(384)
        )
    """)


def create_index(conn):
    conn.execute("""
        CREATE INDEX IF NOT EXISTS paper_chunks_embedding_idx
        ON paper_chunks USING hnsw (embedding vector_cosine_ops)
    """)


def upsert_chunk(conn, chunk_id, arxiv_id, title, chunk_text, published, embedding):
    conn.execute(
        """
        INSERT INTO paper_chunks (chunk_id, arxiv_id, title, chunk_text, published, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding
        """,
        (chunk_id, arxiv_id, title, chunk_text, published, embedding),
    )


if __name__ == "__main__":
    with open(RAW_PAPERS_PATH) as f:
        papers = json.load(f)

    conn = get_connection()

    print("Creating table and extension...")
    create_table(conn)

    print(f"Embedding and inserting {len(papers)} chunks...")
    for paper in papers:
        chunk_id = paper["arxiv_id"]
        chunk_text = f"{paper['title']}\n\n{paper['abstract']}"
        embedding = _model.encode(chunk_text, normalize_embeddings=True)

        upsert_chunk(
            conn,
            chunk_id=chunk_id,
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            chunk_text=chunk_text,
            published=paper["published"],
            embedding=embedding,
        )
        print(f"  inserted {chunk_id}")

    print("Creating HNSW index...")
    create_index(conn)

    conn.close()
    print("Done.")
