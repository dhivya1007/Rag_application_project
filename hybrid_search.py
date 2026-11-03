"""Hybrid (keyword + semantic) search over your downloaded chroma_db.

First time only - build and save the keyword index (a few minutes, needs a few GB RAM):
    python hybrid_search.py --build

Then search:
    python hybrid_search.py "Minneapolis bridge collapse"
    python hybrid_search.py "Minneapolis bridge collapse" --compare     # semantic-only vs hybrid
    python hybrid_search.py "elections" --location "LONDON, England" --n 5
    python hybrid_search.py "Minneapolis bridge collapse" --rerank       # adds a reranked list

Install:  pip install chromadb sentence-transformers bm25s
"""
import argparse
import json
import sys
from pathlib import Path

import bm25s
import chromadb

ROOT = Path(__file__).parent
DB_PATH = str(ROOT / "chroma_db")
COLLECTION = "news_articles"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"      # must match the model used to build the DB
INDEX_DIR = ROOT / "bm25_index"
IDS_FILE = ROOT / "bm25_ids.json"
KW_WEIGHT = 5.0          # keyword list counts 5x in the merge (best setting in evaluate.py so far)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CE = None


def open_collection():
    return chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)


def build_index(col, page=5000):
    ids, docs, off = [], [], 0
    while True:
        r = col.get(include=["documents"], limit=page, offset=off)
        if not r["ids"]:
            break
        ids += r["ids"]
        docs += r["documents"]
        off += len(r["ids"])
        print(f"  read {len(ids)} chunks", end="\r")
    print(f"\ntokenizing {len(ids)} chunks...")
    tokens = bm25s.tokenize(docs, stopwords="en", show_progress=False)
    del docs
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(str(INDEX_DIR))
    IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    print(f"saved keyword index to {INDEX_DIR} and {IDS_FILE.name}")


def load_index():
    if not INDEX_DIR.exists() or not IDS_FILE.exists():
        sys.exit("Keyword index not found. Run once:  python hybrid_search.py --build")
    retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=False)
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    return retriever, ids


def keyword_ids(retriever, ids, query, pool):
    q = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    if not q[0]:
        return []
    res, scores = retriever.retrieve(q, k=min(pool, len(ids)), show_progress=False)
    return [ids[i] for i, s in zip(res[0], scores[0]) if s > 0]


def fuse_articles(sem_ids, kw_ids, k=60, w_sem=1.0, w_kw=1.0):
    """Reciprocal Rank Fusion at ARTICLE level.

    Chunk ids look like '<doc_id>::<n>'. The semantic and keyword search often hit
    different chunks of the same article, so we merge per article, not per chunk.
    Returns (chunk ids to display, score per chunk id, tag per chunk id).
    """
    scores, rep, seen_in = {}, {}, {}
    for name, ranked, weight in (("semantic", sem_ids, w_sem), ("keyword", kw_ids, w_kw)):
        seen = set()
        for cid in ranked:
            doc = cid.rsplit("::", 1)[0]
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + weight / (k + len(seen))
            rep.setdefault(doc, cid)            # semantic's best chunk, else keyword's
            seen_in.setdefault(doc, set()).add(name)
    order = sorted(scores, key=scores.get, reverse=True)
    fused_ids = [rep[d] for d in order]
    score_of = {rep[d]: scores[d] for d in order}
    tag_of = {rep[d]: ("both" if len(seen_in[d]) == 2 else next(iter(seen_in[d]))) for d in order}
    return fused_ids, score_of, tag_of


def show(col, chunk_ids, n, location, tag_of, score_of):
    """Fetch chunk text/metadata, keep each article once, print."""
    if not chunk_ids:
        print("  (no results)")
        return
    got = col.get(ids=chunk_ids, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    seen, shown = set(), 0
    for cid in chunk_ids:
        doc, m = by_id[cid]
        if location and m["location"].lower() != location.lower():
            continue
        if m["doc_id"] in seen:
            continue
        seen.add(m["doc_id"])
        print(f"[{score_of(cid)} | {tag_of(cid)}] {m['location'] or '-'} | {m['title']}\n"
              f"    {doc[:220]}...\n")
        shown += 1
        if shown == n:
            break


def rerank(col, query, fused_ids, top=10):
    """Re-sort the top results with a cross-encoder that reads query + text together."""
    global _CE
    from sentence_transformers import CrossEncoder
    if _CE is None:
        _CE = CrossEncoder(RERANK_MODEL)
    cand = fused_ids[:top]
    got = col.get(ids=cand, include=["documents"])
    text = dict(zip(got["ids"], got["documents"]))
    sc = _CE.predict([(query, text[c]) for c in cand])
    order = sorted(range(len(cand)), key=lambda i: sc[i], reverse=True)
    return [cand[i] for i in order] + fused_ids[top:], {cand[i]: float(sc[i]) for i in order}


def run(query, n, location, compare, pool=60, use_rerank=False):
    from sentence_transformers import SentenceTransformer
    col = open_collection()
    retriever, ids = load_index()
    model = SentenceTransformer(MODEL_NAME)

    qv = model.encode([query], normalize_embeddings=True).tolist()
    sem = col.query(query_embeddings=qv, n_results=pool)
    sem_ids, sem_dist = sem["ids"][0], dict(zip(sem["ids"][0], sem["distances"][0]))
    kw_ids = keyword_ids(retriever, ids, query, pool)
    fused_ids, scores, tags = fuse_articles(sem_ids, kw_ids, w_kw=KW_WEIGHT)

    if compare:
        print("=== SEMANTIC ONLY ===")
        show(col, sem_ids, n, location, lambda c: "semantic", lambda c: f"{1 - sem_dist[c]:.3f}")
        print("=== KEYWORD ONLY ===")
        show(col, kw_ids, n, location, lambda c: "keyword", lambda c: "bm25")
        print("=== HYBRID ===")
    show(col, fused_ids, n, location, lambda c: tags[c], lambda c: f"{scores[c]:.4f}")
    if use_rerank:
        print("=== RERANKED ===")
        ranked_ids, rscores = rerank(col, query, fused_ids)
        show(col, ranked_ids, n, location, lambda c: tags[c], lambda c: f"{rscores.get(c, 0.0):.2f}")


def search_articles(query, n=3, location=None,
                    pool=60, use_rerank=True):

    from sentence_transformers import SentenceTransformer

    col = open_collection()

    # Check whether the database contains articles
    total = col.count()
    if total == 0:
        return []

    retriever, ids = load_index()
    model = SentenceTransformer(MODEL_NAME)

    # 1. Semantic search
    qv = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    sem = col.query(
        query_embeddings=qv,
        n_results=min(pool, total)
    )

    sem_ids = sem["ids"][0]

    # 2. Keyword search
    kw_ids = keyword_ids(
        retriever, ids, query, pool
    )

    # 3. Hybrid fusion
    fused_ids, scores, tags = fuse_articles(
        sem_ids,
        kw_ids,
        w_kw=KW_WEIGHT
    )

    # 4. Reranking
    rerank_scores = {}

    if use_rerank and fused_ids:
        ranked_ids, rerank_scores = rerank(
            col, query, fused_ids
        )
    else:
        ranked_ids = fused_ids

    # 5. Fetch documents and metadata
    got = col.get(
        ids=ranked_ids,
        include=["documents", "metadatas"]
    )

    article_map = {
        cid: (doc, meta)
        for cid, doc, meta in zip(
            got["ids"],
            got["documents"],
            got["metadatas"]
        )
    }

    results = []
    seen = set()

    for cid in ranked_ids:

        if cid not in article_map:
            continue

        doc, meta = article_map[cid]

        article_id = meta.get(
            "doc_id",
            cid.rsplit("::", 1)[0]
        )

        article_location = meta.get("location") or ""

        if location and article_location.lower() != location.lower():
            continue

        if article_id in seen:
            continue

        seen.add(article_id)

        results.append({
            "id": article_id,
            "title": meta.get("title", "Untitled"),
            "location": article_location,
            "text": doc,
            "method": tags.get(cid, "unknown"),
            "rrf_score": scores.get(cid, 0.0),
            "rerank_score": rerank_scores.get(cid)
        })

        if len(results) >= n:
            break

    return results



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--location")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="re-sort the top 20 with a cross-encoder")
    a = ap.parse_args()
    if a.build:
        build_index(open_collection())
    elif a.query:
        run(a.query, a.n, a.location, a.compare, use_rerank=a.rerank)
    else:
        ap.print_help()