import os
import re
from typing import List
import numpy as np

from fastapi import FastAPI, Query
from pydantic import BaseModel

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector, Vector as PgVector

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Quotes RAG (pgvector + local generator)")

# CORS for Vite dev server (Vue at :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # only in dev!
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # "*" cannot be used with credentials=True
)


# ------------------ Config ------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://quotes_user:quotes_pass@localhost:5432/quotes",
)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, same as Lesson 2/3
GEN_MODEL_NAME   = "google/flan-t5-small"                     # lightweight local generator

TOP_K_DEFAULT = 5
MAX_CONTEXT_CHARS = 1200        # guardrail: keep input short for small models
MAX_NEW_TOKENS = 180            # max tokens to generate per answer

# ------------------ DB engine + pgvector ------------------
def _on_connect(dbapi_conn, _):
    register_vector(dbapi_conn)

engine: Engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
event.listen(engine, "connect", _on_connect)

# ------------------ Models ------------------
class Hit(BaseModel):
    id: int
    author: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[Hit]

class RagResponse(BaseModel):
    query: str
    answer: str
    citations: List[Hit]

# ------------------ App + globals ------------------
app = FastAPI(title="Quotes RAG (pgvector + local generator)")

_embedder: SentenceTransformer | None = None
_tok: AutoTokenizer | None = None
_gen: AutoModelForSeq2SeqLM | None = None

# ------------------ Helpers ------------------
def embed_norm(texts: list[str]) -> list[float]:
    """Encode with SBERT and L2-normalize so cosine ≈ inner product."""
    vec = _embedder.encode(texts, normalize_embeddings=False).astype("float32")
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec = vec / norms
    return vec[0].tolist()

def retrieve(q: str, k: int) -> list[Hit]:
    """Nearest-neighbor search in Postgres via pgvector (cosine distance)."""
    qvec = embed_norm([q])
    sql = text("""
        SELECT id, author, text, 1 - (embedding <=> :qvec) AS score
        FROM public.quotes
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :qvec
        LIMIT :k
    """)
    with engine.connect() as con:
        rows = con.execute(sql, {"qvec": PgVector(qvec), "k": k}).mappings().all()
    return [Hit(id=r["id"], author=r["author"], text=r["text"], score=float(r["score"])) for r in rows]

def build_context(hits: list[Hit], max_chars: int) -> str:
    """Concatenate retrieved texts into a compact, numbered context."""
    parts = []
    for i, h in enumerate(hits, start=1):
        seg = f"[{i}] {h.text} — {h.author}"
        parts.append(seg)
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)

def make_prompt(question: str, context: str) -> str:
    """Instruction prompt for FLAN-T5 with explicit citation style."""
    return (
        "You are a helpful assistant. Use ONLY the context to answer the question.\n"
        "Cite supporting quotes using bracketed numbers like [1], [2]. "
        "If the context does not contain the answer, say you cannot answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def generate_answer(prompt: str, max_new_tokens: int) -> str:
    """Run the local seq2seq model to generate an answer."""
    inputs = _tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = _gen.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # deterministic (greedy/beam)
        num_beams=4,              # small beam search for quality
        early_stopping=True,
    )
    return _tok.decode(outputs[0], skip_special_tokens=True).strip()

def ensure_citations(text: str, hits: list[Hit]) -> str:
    """If the model forgot to cite, append top-3 citations."""
    if re.search(r"\[\d+\]", text):
        return text
    add = " " + " ".join(f"[{i}]" for i in range(1, min(3, len(hits)) + 1))
    return text + add

# ------------------ Startup ------------------
@app.on_event("startup")
def startup():
    global _embedder, _tok, _gen
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    _gen = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

# ------------------ Routes ------------------
@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), k: int = Query(TOP_K_DEFAULT, ge=1, le=50)):
    hits = retrieve(q, k)
    return SearchResponse(query=q, results=hits)

@app.get("/ask", response_model=RagResponse)
def ask(
    q: str = Query(..., description="Your question"),
    k: int = Query(TOP_K_DEFAULT, ge=1, le=10, description="Top-k passages"),
    max_ctx_chars: int = Query(MAX_CONTEXT_CHARS, ge=200, le=4000),
    max_new_tokens: int = Query(MAX_NEW_TOKENS, ge=32, le=512),
):
    hits = retrieve(q, k)
    if not hits:
        return RagResponse(query=q, answer="I couldn't find anything relevant to answer.", citations=[])

    ctx = build_context(hits, max_ctx_chars)
    prompt = make_prompt(q, ctx)
    try:
        ans = generate_answer(prompt, max_new_tokens)
    except Exception as e:
        # Fallback: simple extractive stub
        ans = "Here are the most relevant quotes:\n" + "\n".join(f"[{i}] {h.text} — {h.author}"
                                                                 for i, h in enumerate(hits, 1))

    ans = ensure_citations(ans, hits)
    return RagResponse(query=q, answer=ans, citations=hits)
