from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- تحميل config ----------
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

TOP_K = int(config.get("top_k", 7))

# ---------- تحميل chunks ----------
with open(config["chunks_file"], "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ---------- تحميل FAISS index ----------
index = faiss.read_index(config["faiss_index_file"])

# ---------- تحميل embedder ----------
# لازم يكون نفس المودل اللي استخدمتيه وقت بناء الـ embeddings
embedder_name = config.get("embedding_model") or "paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(embedder_name)

# ---------- FastAPI ----------
app = FastAPI(title="RAG QA API")

# CORS (يسمح للواجهة تنادي API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI من نفس السيرفر (أفضل من file://)
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "ok", "ui": "/ui/index.html", "docs": "/docs"}

def _normalize_chunk(item):
    # chunks.json ممكن يكون dict أو string
    if isinstance(item, dict):
        return {
            "text": item.get("text", ""),
            "source": item.get("source", ""),
            "page": item.get("page", "")
        }
    return {"text": str(item), "source": "", "page": ""}

@app.post("/ask")
def ask(q: Question):
    query_vec = np.array(embedder.encode([q.question]), dtype="float32")
    top_k = min(TOP_K, len(chunks))

    D, I = index.search(query_vec, top_k)

    hits = [_normalize_chunk(chunks[int(idx)]) for idx in I[0]]

    # نص جاهز للعرض
    answer_text = "\n\n---\n\n".join(
        [
            f"({h['source']} ص{h['page']})\n{h['text']}".strip()
            if (h["source"] or h["page"])
            else h["text"].strip()
            for h in hits
            if h["text"].strip()
        ]
    )

    return {
        "question": q.question,
        "answer_text": answer_text,
        "sources": hits,
    }
