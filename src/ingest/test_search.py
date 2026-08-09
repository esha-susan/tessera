import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
_model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, k=3):
    conn = psycopg.connect(POSTGRES_URL, autocommit=True)
    register_vector(conn)

    query_embedding = _model.encode(query, normalize_embeddings=True)

    rows = conn.execute(
        """
        SELECT title, 1 - (embedding <=> %s) AS similarity
        FROM paper_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_embedding, query_embedding, k),
    ).fetchall()

    conn.close()
    return rows


if __name__ == "__main__":
    query = "How can a knowledge graph help with multi-hop question answering?"
    results = search(query)

    print(f"Query: {query}\n")
    for title, similarity in results:
        print(f"  {similarity:.3f}  {title}")
