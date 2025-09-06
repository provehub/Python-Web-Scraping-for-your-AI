# --- imports ---
import os  # read DATABASE_URL from env (or use default)
import json  # save metadata for search results
import pathlib  # create a data/ folder for index files
import numpy as np  # vector math & arrays
import pandas as pd  # convenience for reading DB table

from sqlalchemy import create_engine, text  # connect to Postgres and query
from sentence_transformers import SentenceTransformer  # text embeddings
import faiss  # fast vector search

# --- config ---
DATA_DIR = pathlib.Path("data")  # where we store the index + metadata
DATA_DIR.mkdir(exist_ok=True)  # create folder if missing

# Postgres URL (matches your pgAdmin setup; edit as needed)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://quotes_user:quotes_pass@localhost:5432/quotes",
)

INDEX_PATH = DATA_DIR / "quotes.index"  # FAISS index path
META_PATH = DATA_DIR / "meta.json"      # aligned metadata (id,text,author)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & strong baseline
BATCH_SIZE = 256  # embed in batches to avoid RAM spikes
MIN_LEN = 10  # ignore super-short quotes

# --- helpers ---
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length (needed for cosine/IP search)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)  # compute row-wise norms
    norms[norms == 0] = 1.0  # guard against divide-by-zero
    return mat / norms  # return normalized matrix

def load_quotes_df() -> pd.DataFrame:
    """Load quotes table from Postgres."""
    engine = create_engine(DATABASE_URL, future=True)  # create SQLAlchemy engine
    with engine.connect() as con:  # open connection
        df = pd.read_sql(text("SELECT id, text, author FROM quotes"), con)  # pull rows
    df = df.dropna(subset=["text", "author"])  # basic cleanup
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()  # tidy whitespace
    df = df[df["text"].str.len() >= MIN_LEN].reset_index(drop=True)  # filter very short rows
    return df  # ready for embedding

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed a list of texts in batches to a single 2D array."""
    out = []  # chunks of embeddings
    for i in range(0, len(texts), BATCH_SIZE):  # iterate by batch
        batch = texts[i : i + BATCH_SIZE]  # slice current batch
        vecs = model.encode(batch, batch_size=64, show_progress_bar=False, normalize_embeddings=False)  # get vectors
        out.append(vecs.astype("float32"))  # append as float32 for FAISS
    return np.vstack(out)  # stack into one array (N, d)

def main():
    print("Loading data from Postgres…")
    df = load_quotes_df()  # id,text,author
    if df.empty:
        raise SystemExit("No data found in quotes table. Run your scraper first.")

    print(f"Rows to index: {len(df)}")
    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)  # load the SBERT model

    print("Embedding texts…")
    embeddings = embed_texts(model, df["text"].tolist())  # (N, d)

    print("Normalizing embeddings (cosine/IP search)…")
    embeddings = l2_normalize(embeddings)  # unit vectors

    dim = embeddings.shape[1]  # vector dimension d
    print(f"Building FAISS index with dim={dim} and N={len(embeddings)}")
    index = faiss.IndexFlatIP(dim)  # inner-product index (works as cosine if vectors are unit-normalized)
    index.add(embeddings)  # add all vectors

    print("Saving index and metadata…")
    faiss.write_index(index, str(INDEX_PATH))  # save the FAISS index to disk

    meta = [  # store aligned metadata for search results
        {"id": int(i), "text": t, "author": a}
        for i, t, a in zip(df["id"].tolist(), df["text"].tolist(), df["author"].tolist())
    ]
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)  # write metadata JSON

    print(f"Done. Index → {INDEX_PATH.resolve()}")
    print(f"      Meta  → {META_PATH.resolve()}")

if __name__ == "__main__":
    main()  # run the builder
