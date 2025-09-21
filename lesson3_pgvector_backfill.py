import os
import math
import numpy as np
from typing import Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy import event

from sentence_transformers import SentenceTransformer

# register pgvector for psycopg3 connections
from pgvector.psycopg import register_vector, Vector as PgVector

# --- config ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://quotes_user:quotes_pass@localhost:5432/quotes",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
BATCH = 200  # DB batch size
EMB_BATCH = 256  # model batch size

# auto-register pgvector for every new DBAPI connection
def _on_connect(dbapi_conn, _):
    register_vector(dbapi_conn)
engine: Engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
event.listen(engine, "connect", _on_connect)

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def fetch_ids_to_embed() -> list[tuple[int, str]]:
    # get quotes lacking embeddings
    with engine.connect() as con:
        rows = con.execute(text("""
            SELECT id, text
            FROM public.quotes
            WHERE embedding IS NULL
              AND text IS NOT NULL
              AND length(text) >= 10
            ORDER BY id
        """)).fetchall()
    return [(r.id, r.text) for r in rows]

def chunk(it: Iterable, size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    todo = fetch_ids_to_embed()
    if not todo:
        print("Nothing to backfill.")
        return
    print(f"Rows to embed: {len(todo)}")

    for db_batch in chunk(todo, BATCH):
        ids = [i for i, _ in db_batch]
        texts = [t for _, t in db_batch]

        # embed in smaller model batches
        vecs = []
        for sub in chunk(texts, EMB_BATCH):
            v = model.encode(sub, normalize_embeddings=False).astype("float32")
            vecs.append(v)
        emb = np.vstack(vecs)
        emb = l2_normalize(emb)  # store unit vectors for cosine

        with engine.begin() as con:  # transaction
            for i, row_id in enumerate(ids):
                # PgVector wraps the Python list so psycopg knows it's a vector
                con.execute(
                    text("UPDATE public.quotes SET embedding = :e WHERE id = :id"),
                    {"e": PgVector(emb[i].tolist()), "id": row_id},
                )
        print(f"Updated {len(ids)} rows…")

    # optional: analyze to make the planner prefer the ANN index
    with engine.begin() as con:
        con.execute(text("ANALYZE public.quotes"))
    print("Done backfilling embeddings.")

if __name__ == "__main__":
    main()
