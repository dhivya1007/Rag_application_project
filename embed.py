"""Tokenize -> split -> embed -> store in ChromaDB.

Usage:
    python embed_to_chroma.py --corpus corpus.jsonl            # build / update the index
    python embed_to_chroma.py --query "floods in Pakistan"      # test search

Chunks are split by the embedding model's own tokens (not words), so no chunk is
longer than the model's input limit and nothing is silently truncated at embed time.
"""
import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DB_PATH = str(ROOT / "chroma_db")
COLLECTION = "news_articles"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, max 256 tokens
CHUNK_TOKENS = 200       # < 256 so special tokens fit
OVERLAP_TOKENS = 40


def split_by_tokens(text, tokenizer, size=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """Return overlapping text chunks, each at most `size` tokens.

    Uses the tokenizer's character offsets, so chunks are exact slices of the
    original text (no decode artifacts like lowercasing or spaced punctuation).
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=False, verbose=False)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + size >= len(offsets):
            break
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    # We supply our own embeddings, so Chroma needs no embedding function.
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(corpus_path, batch_size=128, limit=None):
    model = SentenceTransformer(MODEL_NAME)
    col = get_collection()
    ids, texts, metas = [], [], []
    total = 0

    def flush():
        nonlocal ids, texts, metas, total
        if not ids:
            return
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
        col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb.tolist())
        total += len(ids)
        print(f"  stored {total} chunks", end="\r")
        ids, texts, metas = [], [], []

    with open(corpus_path, encoding="utf-8") as f:
            for n_docs, line in enumerate(f):
                if limit and n_docs >= limit:
                    break
                d = json.loads(line)
            for i, chunk in enumerate(split_by_tokens(d["text"], model.tokenizer)):
                ids.append(f"{d['id']}::{i}")
                texts.append(chunk)
                metas.append({"doc_id": d["id"], "title": d["title"],
                              "location": d.get("location", ""), "chunk": i})
                if len(ids) >= batch_size:
                    flush()
    flush()
    print(f"\ndone: collection '{COLLECTION}' now holds {col.count()} chunks")


def search(query, n=5, location=None):
    model = SentenceTransformer(MODEL_NAME)
    col = get_collection()
    q = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=n,
                    where={"location": location} if location else None)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"[{1 - dist:.3f}] doc {meta['doc_id']} | {meta['location'] or '-'} | "
              f"{meta['title']}\n    {doc[:200]}...\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--query")
    ap.add_argument("--location", help='optional exact filter, e.g. "LONDON, England"')
    ap.add_argument("--limit", type=int, help="only embed the first N articles")
    args = ap.parse_args()
    if args.query:
        search(args.query, location=args.location)
    else:
        ingest(args.corpus, limit=args.limit)