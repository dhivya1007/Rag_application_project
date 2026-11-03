

"""Data cleaning only: raw CNN/DailyMail CSV -> cleaned CSV.

Usage:
    python clean_data.py --input cnn_dailymail_sample.csv --output cleaned.csv

Output fields: id, title, location, text, reference_summary   (CSV: id, location, article, highlights)
  location - place from the dateline (e.g. "LONDON, England"), kept for filtering/search
  The news-agency tag (e.g. "(CNN) --") is removed from the text and not stored.
"""
import argparse
import csv
import json
import re
from collections import Counter

csv.field_size_limit(10**9)

# "LONDON, England (Reuters) -- text"  |  "(CNN) -- text"  |  "By . Name . PUBLISHED: ..."
DATELINE = re.compile(
    r"^\s*(?:(?P<loc>[A-Z][A-Za-z0-9 .,'/&-]{1,60}?)\s*)?"
    r"\((?P<src>[A-Za-z][A-Za-z .&]{1,30})\)\s*(?:--|—|–|-|:)?\s*")
LATE_DATELINE = re.compile(
    r"(?<=[.!?\"'])\s+(?:(?P<loc>[A-Z][A-Za-z0-9 .,'/&-]{1,60}?)\s*)?"
    r"\((?P<src>CNN|Reuters|AP|AFP|BBC)\)\s*(?:--|—|–)\s*")
DAILYMAIL = re.compile(r"^By\s+\.\s+.*?(PUBLISHED|UPDATED):.*?\d{4}\s*\.\s*(\|\s*)?", re.I | re.S)
HTML = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACES = re.compile(r"\s+")
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})


def clean_article(raw: str):
    text = HTML.sub(" ", raw)
    text = DAILYMAIL.sub("", text)
    location = ""
    m = DATELINE.match(text)
    if m:
        location = (m.group("loc") or "").strip(" ,")
        text = text[m.end():]
    else:  # dateline after an intro / "Editor's note" paragraph
        m = LATE_DATELINE.search(text[:1200])
        if m:
            location = (m.group("loc") or "").strip(" ,")
            text = text[:m.start()] + " " + text[m.end():]
    text = URL.sub("", text)
    text = SPACES.sub(" ", text.translate(QUOTES)).strip()
    return location, text


def make_title(text: str, max_words: int = 12) -> str:
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].split()
    return " ".join(first[:max_words]) + ("..." if len(first) > max_words else "")


def clean_plain(raw: str) -> str:
    return SPACES.sub(" ", HTML.sub(" ", raw).translate(QUOTES)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,default="cnn_dailymail_sample.csv")
    ap.add_argument("--output", default="cleaned.csv")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=3000)
    a = ap.parse_args()

    stats, seen, out = Counter(), set(), []
    with open(a.input, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stats["read"] += 1
            loc, text = clean_article(row["article"])
            n = len(text.split())
            if n < a.min_words:
                stats["dropped_too_short"] += 1
                continue
            if n > a.max_words:
                stats["dropped_too_long"] += 1
                continue
            key = text[:200].lower()
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)
            stats["with_location"] += bool(loc)
            out.append({"id": row.get("id", stats["read"]), "location": loc,
                        "article": text, "highlights": clean_plain(row.get("highlights", ""))})

    if a.output.endswith(".jsonl"):
        with open(a.output, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps({
                    "id": str(r["id"]), "title": make_title(r["article"]),
                    "location": r["location"],
                    "text": r["article"], "reference_summary": r["highlights"]},
                    ensure_ascii=False) + "\n")
    else:
        with open(a.output, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "location", "article", "highlights"])
            w.writeheader()
            w.writerows(out)

    print(f"read={stats['read']} kept={len(out)}")
    for k in ("dropped_too_short", "dropped_too_long", "dropped_duplicate", "with_location"):
        print(f"{k}={stats[k]}")


if __name__ == "__main__":
    main()

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


def ingest(corpus_path, batch_size=128):
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
        for line in f:
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
    args = ap.parse_args()
    if args.query:
        search(args.query, location=args.location)
    else:
        ingest(args.corpus)