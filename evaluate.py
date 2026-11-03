"""Retrieval accuracy: semantic-only vs keyword-only vs hybrid, on your full chroma_db.

Each test query is built from a real article's reference summary; a hit means the
search returned THAT article. Needs the keyword index first (python hybrid_search.py --build).

    python evaluate.py                    # 200 random articles
    python evaluate.py --n 500 --seed 7
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import hybrid_search as hs

ROOT = Path(__file__).parent


def article_order(chunk_ids):
    """Chunk ids ('<doc_id>::<n>') -> article ids, best first, duplicates removed."""
    seen, out = set(), []
    for c in chunk_ids:
        d = c.rsplit("::", 1)[0]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def rank_of(target, order):
    return order.index(target) + 1 if target in order else None


def metrics(ranks):
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1.0 / r for r in ranks if r and r <= 10) / n
    return hit(1), hit(5), hit(10), mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--rerank", action="store_true", help="also test hybrid + cross-encoder reranker (slower)")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    col = hs.open_collection()
    retriever, ids = hs.load_index()
    model = SentenceTransformer(hs.MODEL_NAME)

    with open(a.corpus, encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(a.seed)
    docs = [json.loads(lines[i]) for i in rng.sample(range(len(lines)), min(a.n, len(lines)))]
    docs = [d for d in docs if d.get("reference_summary", "").strip()]
    print(f"{len(docs)} test articles out of {len(lines)} in the corpus\n")

    variants = {
        "full summary": lambda s: s,
        "short query (first 8 words)": lambda s: " ".join(s.split()[:8]),
    }
    rows = []
    for vname, make_q in variants.items():
        queries = [make_q(d["reference_summary"]) for d in docs]
        t0 = time.time()
        qv = model.encode(queries, batch_size=64, normalize_embeddings=True).tolist()
        sem_all = col.query(query_embeddings=qv, n_results=a.pool)["ids"]
        weights = [1, 2, 3, 5]
        ranks = {"semantic only": [], "keyword only": []}
        ranks.update({f"hybrid (keyword x{w})": [] for w in weights})
        if a.rerank:
            ranks["hybrid x5 + reranker"] = []
        for d, q, sem_ids in zip(docs, queries, sem_all):
            kw_ids = hs.keyword_ids(retriever, ids, q, a.pool)
            ranks["semantic only"].append(rank_of(d["id"], article_order(sem_ids)))
            ranks["keyword only"].append(rank_of(d["id"], article_order(kw_ids)))
            for w in weights:
                fused_ids, _, _ = hs.fuse_articles(sem_ids, kw_ids, w_kw=w)
                ranks[f"hybrid (keyword x{w})"].append(rank_of(d["id"], article_order(fused_ids)))
            if a.rerank:
                fused5, _, _ = hs.fuse_articles(sem_ids, kw_ids, w_kw=hs.KW_WEIGHT)
                reranked, _ = hs.rerank(col, q, fused5)
                ranks["hybrid x5 + reranker"].append(rank_of(d["id"], article_order(reranked)))
        print(f"== queries: {vname}   ({time.time() - t0:.0f}s) ==")
        print(f"{'method':<24}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}")
        for m, rk in ranks.items():
            h1, h5, h10, mrr = metrics(rk)
            print(f"{m:<24}{h1:>8.1%}{h5:>8.1%}{h10:>8.1%}{mrr:>8.3f}")
            rows.append([vname, m, round(h1, 4), round(h5, 4), round(h10, 4), round(mrr, 4), len(rk)])
        print()

    out = ROOT / "eval_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_type", "method", "hit@1", "hit@5", "hit@10", "MRR", "n_queries"])
        w.writerows(rows)
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()