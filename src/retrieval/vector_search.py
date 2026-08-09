import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
_model = SentenceTransformer("all-MiniLM-L6-v2")


def vector_search(query: str, k: int = 5) -> list[dict]:
    conn = psycopg.connect(POSTGRES_URL, autocommit=True)
    register_vector(conn)

    query_embedding = _model.encode(query, normalize_embeddings=True)

    rows = conn.execute(
        """
        SELECT chunk_id, title, chunk_text, 1 - (embedding <=> %s) AS similarity
        FROM paper_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_embedding, query_embedding, k),
    ).fetchall()

    conn.close()

    return [
        {
            "chunk_id": row[0],
            "title": row[1],
            "chunk_text": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]
