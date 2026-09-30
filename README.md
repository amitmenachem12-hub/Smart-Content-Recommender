# Smart Content Recommender
### *Stop Scrolling. Start Watching.*

A reactive movie and TV show recommendation engine that understands what you mean, not just what you type. Instead of rigid genre filters, it translates natural-language queries into semantic vectors, then refines those vectors in real time based on your feedback — using the classical **Rocchio relevance feedback** algorithm from Information Retrieval.

---

## Overview

Most recommendation systems match keywords or rely on collaborative filtering ("users like you also watched…"). This project takes a different approach:

1. **You describe a feeling** — *"something cozy for a Sunday evening"* or *"מותח פשע ישראלי"* (Israeli crime thriller in Hebrew)
2. **The engine finds semantic neighbours** in a 384-dimensional embedding space
3. **You rate a handful of candidates** — a few stars per title, no full watch required
4. **The query vector shifts** — pulled toward concepts you liked, pushed away from concepts you didn't — and a personalized ranked list emerges

The result is a two-round recommendation loop that converges on your actual taste rather than guessing it.

---

## System Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  Stage 0: Query  ──►  Stage 1: Rate Candidates  ──►  Stage 2: Results │
└────────┬─────────────────────┬──────────────────────────┬────────┘
         │                     │                          │
         ▼                     ▼                          ▼
  ┌─────────────┐    ┌──────────────────┐     ┌──────────────────────┐
  │ Query Text  │    │ User Ratings     │     │  Final Ranked List   │
  │  (free text)│    │ (1–5 stars each) │     │  + Provider Filter   │
  └──────┬──────┘    └────────┬─────────┘     │  + YouTube Trailers  │
         │                    │               └──────────────────────┘
         ▼                    ▼
  ┌─────────────────────────────────────────────┐
  │              search_engine.py               │
  │                                             │
  │  1. Encode query → 384-dim vector           │
  │     (paraphrase-multilingual-MiniLM-L12-v2) │
  │  2. Query ChromaDB (HNSW, cosine sim)       │
  │  3. Smart filters: media type, genre,       │
  │     language, safe-search, deduplication    │
  │  4. Return pool (30) + top-k (10) + vector  │
  └──────────────────────────┬──────────────────┘
                             │
                        query_vector
                             │
                             ▼
  ┌─────────────────────────────────────────────┐
  │           relevance_feedback.py             │
  │                                             │
  │  Rocchio Algorithm:                         │
  │                                             │
  │  V_new = α·V_query                          │
  │        + β · mean(V_liked)                  │
  │        − γ · mean(V_disliked)               │
  │                                             │
  │  α=1.0  β=0.7  γ=0.3                       │
  │  Ratings ≥4 → positive   ≤2 → negative     │
  │  V_new L2-normalised for cosine compat.     │
  └──────────────────────────┬──────────────────┘
                             │
                        shifted vector
                             │
                             ▼
  ┌─────────────────────────────────────────────┐
  │          ChromaDB (Local SQLite/HNSW)       │
  │                                             │
  │  ~4 000 titles (movies + TV, EN + HE)       │
  │  Stored embeddings: title + overview +      │
  │  genres + media_type + language             │
  │  Cosine similarity re-ranked against V_new  │
  └──────────────────────────┬──────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────┐
  │             tmdb_client.py                  │
  │                                             │
  │  Enrichment (concurrent, ThreadPoolExec):   │
  │  • Watch providers per title (IL / US)      │
  │  • YouTube trailer key                      │
  │  • Content certification (age rating)       │
  └─────────────────────────────────────────────┘
```

### The Two-Round Loop in Detail

**Round 1 — Semantic Search**

The user's natural-language query is encoded by `paraphrase-multilingual-MiniLM-L12-v2` into a 384-dimensional vector. The engine queries ChromaDB for the closest neighbours by cosine similarity, then applies Python-side filters:

- **Genre boosting** — keywords like *"sci-fi"* or *"מצחיק"* (funny) are mapped to TMDB genre IDs and boost matching candidates.
- **Media-type detection** — regex detects *"show / series / סדרה"* vs *"movie / film / סרט"*, filtering the result set accordingly.
- **Lighthearted penalty** — when the query signals lighthearted intent (*"funny"*, *"relaxing"*), titles containing dark semantic cues (*"murder"*, *"trauma"*) are down-weighted.
- **Deduplication** — sequels are swapped for originals; titles with >50 % overlapping words are collapsed to one.

Ten candidates are shown to the user as rating cards.

**Round 2 — Rocchio Vector Shift**

After the user rates candidates, `apply_user_feedback()` computes a new query vector using the Rocchio formula:

```
V_new = α · V_query  +  β · mean(V_liked)  −  γ · mean(V_disliked)
```

| Parameter | Value | Role |
|---|---|---|
| α (alpha) | 1.0 | Preserves original query intent |
| β (beta) | 0.7 | Moves toward liked semantic concepts |
| γ (gamma) | 0.3 | Moves away from disliked concepts (conservative to avoid erratic jumps) |

`V_new` is L2-normalised so it remains compatible with ChromaDB's cosine-distance metric. Phase-1 candidates are excluded from the final list, ensuring the user always sees fresh results.

---

## Features

| Feature | Details |
|---|---|
| **Semantic Search** | Free-text queries mapped to embeddings — no keyword matching required |
| **Relevance Feedback** | Rocchio algorithm shifts the query vector based on star ratings, not re-filtering |
| **Bilingual (EN / HE)** | Handles English and Hebrew queries natively; full RTL UI for Hebrew |
| **Streaming Provider Filter** | Real-time multiselect filter by Netflix, Disney+, Apple TV+ and others via TMDB watch providers |
| **YouTube Trailers** | Embedded trailers fetched from TMDB and rendered inline |
| **Safe Search Toggle** | Filters out R / NC-17 / TV-MA content on demand |
| **Smart Deduplication** | Sequel-to-original swaps and word-overlap collapsing for diverse results |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI & App State | Python 3.11 · Streamlit 1.64 |
| Vector Database | ChromaDB 1.5.9 (Local SQLite + HNSW index, cosine similarity) |
| Embeddings | `sentence-transformers` 6.1 — `paraphrase-multilingual-MiniLM-L12-v2` |
| External Data | TMDB API v3 (metadata, watch providers, trailers) |
| Vector Math | NumPy 2.4 |
| Config | `python-dotenv` |

---

## Project Structure

```
smart-content-recommender/
├── src/
│   ├── app.py                      # Streamlit entry point — 3-stage UI flow
│   ├── search_engine.py            # Semantic search, genre boosting, deduplication
│   ├── relevance_feedback.py       # Rocchio algorithm implementation
│   ├── embeddings.py               # Embedding generation + ChromaDB build script
│   ├── tmdb_client.py              # TMDB REST API client (metadata + providers)
│   └── translations.py             # EN / HE UI strings
├── tests/
│   └── test_relevance_feedback.py  # 16+ unit tests for Rocchio correctness
├── data/
│   ├── movies_cache.json           # Raw TMDB metadata (~2 MB, gitignored)
│   └── chroma_db/                  # Persistent ChromaDB vector index (gitignored)
├── .env                            # API keys — never committed
├── requirements.txt
└── CLAUDE.md
```

---

## Installation & Local Setup

### Prerequisites

- Python 3.11+
- A free [TMDB API](https://www.themoviedb.org/settings/api) account (takes ~2 minutes)

### 1. Clone the repository

```bash
git clone https://github.com/amitmenachem12-hub/Smart-Content-Recommender.git
cd Smart-Content-Recommender
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
TMDB_ACCESS_TOKEN=your_tmdb_read_access_token_here
```

> **Where to find this:** TMDB Dashboard → Settings → API → *API Read Access Token (v4 auth)*.
> The token begins with `eyJ...` and is used as a Bearer token — it is not the same as the shorter API Key (v3).

### 5. Build the vector database

This fetches ~4 000 titles from TMDB, generates embeddings, and writes the ChromaDB index to `data/chroma_db/`. Run once; takes roughly 5–10 minutes depending on your connection and CPU.

```bash
python src/tmdb_client.py   # fetch + cache metadata  (~2 min)
python src/embeddings.py    # generate embeddings + build ChromaDB index  (~5 min)
```

### 6. Launch the app

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage Example

**Search query:** *"a gripping Israeli crime thriller"*

1. The engine encodes the query and returns 10 candidate titles ranked by cosine similarity.
2. You rate each one — 5 stars for *Tehran*, 1 star for a rom-com that slipped in.
3. Rocchio shifts the vector: toward spy-thriller semantic space, away from romantic-comedy space.
4. The final 10 recommendations reflect your refined intent — filter instantly by whichever streaming service you subscribe to.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers Rocchio correctness using 3-dimensional mock vectors — no sentence-transformer required, so tests run fast and offline.

---

## License

[MIT](LICENSE)

---

> "This product uses the TMDB API but is not endorsed or certified by TMDB."
