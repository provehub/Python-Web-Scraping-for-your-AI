# --- imports ---
import json  # load metadata list (id,text,author)
import pathlib  # build portable paths
from typing import List  # type hints for response model

import faiss  # load/search the FAISS index
import numpy as np  # vector math
from fastapi import FastAPI, Query  # minimal API
from pydantic import BaseModel  # response schema
from sentence_transformers import SentenceTransformer  # same model used to build index

# --- config ---
DATA_DIR = pathlib.Path("data")  # folder with index/meta
INDEX_PATH = DATA_DIR / "quotes.index"  # FAISS index file
META_PATH = DATA_DIR / "meta.json"      # aligned metadata file
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # must match builder

# --- utils ---
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise unit normalization (for cosine/IP search)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

# --- API models ---
class SearchHit(BaseModel):
    id: int  # DB id (quotes.id)
    author: str  # author name
    text: str  # quote text
    score: float  # similarity score (cosine ∈ [0,1] since we normalized)

class SearchResponse(BaseModel):
    query: str  # the input query
    k: int  # how many results returned
    results: List[SearchHit]  # top-k hits

# --- app init ---
app = FastAPI(title="Quotes Semantic Search")  # create FastAPI app

# Globals populated at startup
_index = None  # FAISS index in memory
_meta = []  # list of dicts (id,text,author)
_model = None  # SentenceTransformer model

@app.on_event("startup")
def load_assets():
    """Load FAISS index, metadata, and the embedding model once."""
    global _index, _meta, _model  # modify module-level variables
    if not INDEX_PATH.exists() or not META_PATH.exists():  # ensure files exist
        raise RuntimeError("Index or metadata missing. Run lesson2_build_index.py first.")
    _index = faiss.read_index(str(INDEX_PATH))  # read FAISS index from disk
    _meta = json.loads(META_PATH.read_text(encoding="utf-8"))  # load metadata list
    _model = SentenceTransformer(MODEL_NAME)  # load same model as builder

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1, description="Search query text"),
           k: int = Query(5, ge=1, le=50, description="Top-k results")):
    """Return the top-k most similar quotes to the query."""
    # Encode the query text to a vector (1, d)
    vec = _model.encode([q], normalize_embeddings=False).astype("float32")
    vec = l2_normalize(vec)  # unit-normalize for cosine/IP search

    # Search FAISS index: scores (1, k) and indices (1, k)
    scores, idxs = _index.search(vec, min(k, len(_meta)))  # cap k at index size

    # Build response hits by looking up metadata
    hits = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i == -1:  # FAISS returns -1 for empty results; guard if the index were empty
            continue
        m = _meta[i]  # aligned metadata
        # Convert inner-product score to cosine-like in [0, 1] (already unit vectors)
        hits.append(SearchHit(id=m["id"], author=m["author"], text=m["text"], score=float(score)))

    return SearchResponse(query=q, k=len(hits), results=hits)  # FastAPI returns JSON
