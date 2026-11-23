"""Retrieval layer: stores the resume, job description and question bank in Chroma
and returns the chunks most relevant to a query."""

import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(DB_PATH))
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)


def _collection(name):
    # Chroma's default embedding model (all-MiniLM-L6-v2) runs locally for free.
    return _client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _read_text(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def ingest_candidate(resume_path, jd_path=None):
    """Chunk the resume (and optional job description) and store them as embeddings."""
    if "candidate" in [c.name for c in _client.list_collections()]:
        _client.delete_collection("candidate")
    col = _collection("candidate")
    total = 0
    for doc_type, path in (("resume", resume_path), ("job_description", jd_path)):
        if not path:
            continue
        chunks = _splitter.split_text(_read_text(path))
        col.add(
            ids=[f"{doc_type}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"type": doc_type, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"  {doc_type}: {len(chunks)} chunks from {Path(path).name}")
    return total


def ingest_question_bank(path):
    """Store every question with its topic so it can be searched by meaning or filtered by topic."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if "questions" in [c.name for c in _client.list_collections()]:
        _client.delete_collection("questions")
    col = _collection("questions")
    col.add(
        ids=[f"q-{i}" for i in range(len(items))],
        documents=[f"[{q['topic']}] {q['question']}" for q in items],
        metadatas=[{"topic": q["topic"], "key_points": "; ".join(q["key_points"])} for q in items],
    )
    print(f"  question bank: {len(items)} questions")
    return len(items)


def search(collection, query, k=3, where=None, exclude_ids=()):
    """Return the top-k chunks as a list of (id, text, metadata), skipping excluded ids."""
    res = _collection(collection).query(query_texts=[query], n_results=k + len(exclude_ids), where=where)
    hits = [(i, d, m) for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
            if i not in exclude_ids]
    return hits[:k]
