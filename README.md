# Rag_application_project
Document search and summarization using hybrid retrieval and LLM API.


# News RAG: Hybrid Search and Summarization

A Retrieval-Augmented Generation (RAG) system over CNN/DailyMail news articles. It stores cleaned articles in ChromaDB, retrieves them with hybrid keyword + semantic search (re-ranked with a cross-encoder), and uses an LLM served by Groq to answer a query or summarize the results at an adjustable length. If the retrieved articles don't answer the query (for example, "floods in Pakistan" when the corpus has no such story), the system says so instead of inventing an answer. The interface is built with Streamlit.

## How it works

```
cnn_dailymail_sample.csv
        |  clean_data.py
        v
corpus.jsonl  (cleaned articles)
        |  embed.py  (split into 200-token chunks, embed, store)
        v
chroma_db/  (ChromaDB, cosine HNSW)     bm25_index/  (keyword index)
        |                                      |
        +------------- hybrid_search.py -------+
          semantic + BM25 (keyword weight 5 : semantic 1)
          -> article-level Reciprocal Rank Fusion
          -> cross-encoder re-ranking
                              |
                        top N articles
                              |  summarizer.py  (Groq LLM)
                              v
              answer / summary  (or "No relevant information found")
                              |
                         app.py (Streamlit)
```

| Step | File | What it does |
|---|---|---|
| 0. Download | `datasetloading.py` | Downloads the CNN/DailyMail sample from Hugging Face and saves it as `cnn_dailymail_sample.csv`. |
| 1. Clean | `datacleaning.py` | Removes HTML, URLs and promo text. Parses the dateline into a `location` field, drops duplicates and articles under 80 words, and writes `corpus.jsonl`. |
| 2. Embed and store | `embed.py` | Splits each article into 200-token chunks (40 overlap) using the model's own tokenizer, embeds them with `all-MiniLM-L6-v2` (384 dimensions), and stores them in ChromaDB with metadata (`doc_id`, `title`, `location`, `chunk`). |
| 3. Search | `hybrid_search.py` | Runs semantic search (ChromaDB) and keyword search (BM25), merges them with a keyword weight of 5 to a semantic weight of 1, then re-ranks the top results with a cross-encoder. |
| 4. Generate | `summarizer.py` | Builds a prompt from the retrieved articles and calls the LLM through the Groq API. |
| 5. Evaluate | `evaluate.py` | Measures retrieval accuracy for semantic, keyword and hybrid search. |
| 6. Interface | `app.py` | Streamlit app: enter a query, choose the summary length and number of articles, and see the answer with its source articles. |

## Design decisions

- **Hybrid search, weighted towards keywords.** News queries are full of names, places and events ("Minneapolis bridge collapse"), and in testing keyword (BM25) search worked better than semantic search on this data. Semantic search is still kept, because it catches paraphrases and related wording that keyword search misses. The two result lists are merged with Reciprocal Rank Fusion, with the keyword list weighted **5** and the semantic list weighted **1**: `score = 5 / (60 + rank_kw) + 1 / (60 + rank_sem)`. The weight (`KW_WEIGHT` in `hybrid_search.py`) was chosen by running `evaluate.py` with weights 1, 2, 3 and 5.
- **Fusion at article level.** The two searches often hit different chunks of the same article, so results are merged per article, not per chunk.
- **Cross-encoder re-ranking.** After fusion, the top results are re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which reads the query and the article text together and so judges relevance more carefully than the first-stage search. It is slower, so it is only applied to the top candidates. It can be turned off with `use_rerank=False`.
- **Chunking by tokens, not words.** `all-MiniLM-L6-v2` reads at most 256 tokens. Chunks of 200 tokens leave room for special tokens, so nothing is silently cut off when embedding.
- **No-information handling.** The prompt tells the model to use only the supplied articles, and to reply with the exact sentence `No relevant information found in the documents.` when they don't answer the query. Because the sentence is fixed, the code can detect it.
- **Adjustable summary length.** `short` (2-3 sentences), `medium` (one paragraph) or `long` (about three paragraphs), set in the interface or with `--length`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use: source .venv/bin/activate on macOS/Linux)

pip install -r requirements.txt
```

(Main packages: `chromadb`, `sentence-transformers`, `bm25s`, `groq`, `python-dotenv`, `streamlit`.)

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_key_here
```

## Usage

### 1. Prepare the data (once)

```bash
python datasetloading.py                  # downloads the data -> cnn_dailymail_sample.csv
python datacleaning.py                    # cnn_dailymail_sample.csv -> corpus.jsonl
python embed.py --corpus corpus.jsonl     # builds chroma_db/ (slow on CPU; a GPU is much faster)
python hybrid_search.py --build           # builds the keyword index (bm25_index/, bm25_ids.json)
```

### 2. Search from the command line

```bash
python hybrid_search.py "Minneapolis bridge collapse"
python hybrid_search.py "Minneapolis bridge collapse" --compare   # semantic vs keyword vs hybrid
python hybrid_search.py "Minneapolis bridge collapse" --rerank    # adds the re-ranked list
python hybrid_search.py "elections" --location "LONDON, England" --n 5
```

### 3. Run the interface

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The first search is slow because the embedding model, keyword index and re-ranker load then; later searches are fast.

## Evaluation

`evaluate.py` picks random articles and uses their reference summaries (full text, and the first 8 words) as queries. A hit means the search returned that article. It reports hit@1, hit@5, hit@10 and MRR for semantic only, keyword only, and hybrid at several keyword weights.

```bash
python evaluate.py --n 500
python evaluate.py --n 500 --rerank      # also tests hybrid + the cross-encoder (slower)
```

Results are saved to `eval_results.csv`. Paste your own numbers here after running it:

| Method | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Semantic only | | | | |
| Keyword only | | | | |
| Hybrid (keyword x5) | | | | |
| Hybrid (keyword x5) + re-ranker | | | | |

Note: the test queries are built from each article's own summary, which favors keyword overlap. A re-ranker may score differently on real user queries than on this benchmark, so report both honestly.

Summary quality is not scored automatically yet. ROUGE against the dataset's reference summaries is the natural next step.

## Scaling notes

- **Current size:** the 50,000-row corpus (47,532 articles after cleaning) becomes about 249,000 chunks, around 1.2 GB on disk. ChromaDB with a local HNSW index handles this on one machine.
- **Larger corpora:** the vector store can move to pgvector (keeps vectors next to the metadata in PostgreSQL, SQL filtering) or FAISS (fastest in-memory search, but no built-in metadata storage). Keyword search can move to OpenSearch/Elasticsearch.
- **Embedding model:** if retrieval needs to improve, a model with a longer input (for example BAAI `bge`, 512 tokens) gives fewer, larger chunks. The database must be rebuilt when the model changes.
- **Latency:** the cross-encoder is the slowest part of a query, so it is limited to the top candidates. Models are loaded once and reused.

## Limitations

- The corpus is a sample of about 50,000 CNN/DailyMail articles (47,532 after cleaning), so many real-world topics are not covered (that is what the no-information behavior is for).
- The `location` filter needs an exact match with the dateline text (for example `LONDON, England`).
- Answers depend on the Groq API being available and on a valid API key.

## Project structure

```
rag_Application/
  datasetloading.py          downloads the raw dataset
  cnn_dailymail_sample.csv   raw data
  datacleaning.py            cleans the raw data
  corpus.jsonl               cleaned articles
  embed.py                   chunks, embeds and stores in ChromaDB
  chroma_db/                 vector database (chroma_db.zip is a backup copy)
  hybrid_search.py           keyword + semantic search, fusion, re-ranking
  bm25_index/, bm25_ids.json keyword index
  summarizer.py              LLM answer / summary (Groq)
  evaluate.py                retrieval evaluation
  eval_results.csv           evaluation results
  app.py                     Streamlit interface
  requirements.txt           Python dependencies
  .env                       GROQ_API_KEY
  README.md
```
