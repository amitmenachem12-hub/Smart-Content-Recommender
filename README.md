# Smart Content Recommender
### *Stop Scrolling. Start Watching.*

A production-grade, hybrid movie and TV recommendation engine that understands natural-language intent, refines it through inline user feedback, and fuses two independent ranking signals into a single personalized list — all inside a reactive Streamlit UI.

---

## Overview

Most recommendation systems pick one strategy: keyword matching, collaborative filtering, or popularity sorting. This project combines all three into a two-round pipeline that converges on your actual taste rather than guessing it.

1. **You describe a feeling** — *"something funny and cozy to watch with friends"* or *"מותח פשע ישראלי"* (Israeli crime thriller in Hebrew)
2. **The engine finds semantic neighbours** in a 384-dimensional embedding space via ChromaDB
3. **You rate a handful of candidates** — 1 to 5 stars per title, inline
4. **Two things happen simultaneously:**
   - The query vector is *shifted* by the Rocchio algorithm toward liked concepts and away from disliked ones
   - The TMDB `/discover` endpoint is queried with your taste profile to surface widely-known cultural peers (e.g. HIMYM and New Girl when you rate Friends 5 stars)
5. **Both result lists are fused** via Reciprocal Rank Fusion (RRF) and filtered by a dynamic genre blacklist before the final ranked list is shown

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI  (app.py)                         │
│                                                                             │
│   Stage 0: Query Input  ──►  Stage 1: Rate Candidates  ──►  Stage 2: Recs  │
└──────────┬──────────────────────────┬───────────────────────────┬───────────┘
           │                          │                           │
           ▼                          ▼                           ▼
    Free-text query            1–5 star ratings           Final ranked list
                               per candidate              + Provider filter
                                                          + YouTube trailers
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────────────────────────────────┐
│   semantic_search()  │   │               Phase 2 Hybrid Engine              │
│   search_engine.py   │   │               search_engine.py                   │
│                      │   │                                                  │
│  • Encode query      │   │  ┌─────────────────────────────────────────────┐ │
│    → 384-dim vector  │   │  │  relevance_feedback.py  (Rocchio)           │ │
│  • ChromaDB HNSW     │   │  │                                             │ │
│    cosine search     │   │  │  V_new = α·V_query                          │ │
│  • Hard filters:     │   │  │        + β · Σ(wᵢ·V_liked) / |pos|         │ │
│    media type,       │   │  │        − γ · mean(V_disliked)               │ │
│    genre, language,  │   │  │                                             │ │
│    safe-search       │   │  │  α=1.0  β=0.8  γ=0.3  (L2-normalised)     │ │
│  • Deduplication     │   │  └──────────────┬──────────────────────────────┘ │
│    (sequel swap,     │   │                 │ shifted vector                  │
│     title overlap)   │   │                 ▼                                │
│                      │   │  ┌──────────────────────────────────────────────┐ │
│  Returns:            │   │  │  ChromaDB vector search  (Signal A)         │ │
│  • 24 candidates     │   │  │  Top-120 cosine neighbours of V_new         │ │
│  • query_vector      │   │  └──────────────┬──────────────────────────────┘ │
└──────────────────────┘   │                 │                                │
                           │                 │ chroma_results                 │
                           │                 ▼                                │
                           │  ┌──────────────────────────────────────────────┐ │
                           │  │  TMDB /discover  (Signal B)                 │ │
                           │  │                                              │ │
                           │  │  build_genre_profile(anchor_items):         │ │
                           │  │  • dominant_genre  = most-frequent genre    │ │
                           │  │    across highly-rated Phase 1 items        │ │
                           │  │  • banned_genres   = TONE_SHIFTING_GENRES   │ │
                           │  │                      minus anchor genres     │ │
                           │  │                                              │ │
                           │  │  /discover/tv  (TV anchors only)            │ │
                           │  │    with_genres   = dominant_genre_id (AND)  │ │
                           │  │    without_genres = banned_genre_ids        │ │
                           │  │    sort_by        = popularity.desc         │ │
                           │  │    vote_count.gte = 1500                    │ │
                           │  │                                              │ │
                           │  │  /discover/movie  (Movie anchors only)      │ │
                           │  │  → look up returned IDs in ChromaDB index   │ │
                           │  └──────────────┬──────────────────────────────┘ │
                           │                 │ tmdb_results                   │
                           │                 ▼                                │
                           │  ┌──────────────────────────────────────────────┐ │
                           │  │  filter_candidates()  (genre gate)          │ │
                           │  │  • Reject if NO overlap with anchor genres  │ │
                           │  │  • Reject if ANY banned genre present       │ │
                           │  └──────────────┬──────────────────────────────┘ │
                           │                 │ filtered_chroma, filtered_tmdb │
                           │                 ▼                                │
                           │  ┌──────────────────────────────────────────────┐ │
                           │  │  reciprocal_rank_fusion(list_A, list_B, k=60)│ │
                           │  │                                              │ │
                           │  │  score(item) = Σ  1 / (60 + rank_in_list)  │ │
                           │  │  Items in both lists rank highest           │ │
                           │  └──────────────┬──────────────────────────────┘ │
                           │                 │ fused & ranked                 │
                           └─────────────────┼────────────────────────────────┘
                                             │
                                             ▼
                           ┌──────────────────────────────────────────────────┐
                           │  tmdb_client.py  (concurrent enrichment)        │
                           │  • IL watch providers  (Netflix, Disney+, …)    │
                           │  • YouTube trailer key                          │
                           │  • EN + HE titles & overviews                   │
                           │  • Content certification (age rating)           │
                           └──────────────────────────────────────────────────┘
```

---

## Key Technical Features

### 1. Two-Stage Reactive UI

The Streamlit app runs as a three-state machine:

- **Stage 0 — Search:** Free-text query input with an optional Family-Friendly toggle. Submitting triggers a full semantic search and pre-enriches the candidate pool in parallel using `concurrent.futures.ThreadPoolExecutor`.
- **Stage 1 — Rate:** 24 candidates are displayed in a responsive grid with poster images, synopses, and inline YouTube trailers. The user assigns 1–5 stars to any number of titles.
- **Stage 2 — Results:** Personalized recommendations rendered as enriched cards. A real-time streaming-provider multiselect filter (Netflix, Prime, Disney+, etc.) narrows results without a page reload.

### 2. Hybrid Recommendation Engine

Phase 2 combines two independent ranking signals:

| Signal | Source | Mechanism |
|---|---|---|
| **Semantic** | ChromaDB + Sentence-Transformers | Cosine similarity of Rocchio-shifted query vector against 384-dim title embeddings |
| **Popularity** | TMDB `/discover` endpoint | Genre-filtered, vote-count-gated popularity sort — surfaces cultural landmarks that semantic vectors miss |

The two result lists are kept separate until the final RRF merge step, so neither signal can fully override the other.

### 3. Dynamic Genre Blacklist + Single-Dominant-Genre Filtering

The engine builds a personalised genre fence from the user's own ratings rather than a static rule:

**`build_genre_profile(anchor_items)`** — extracts all genres from Phase 1 items rated ≥ 4 stars:
- `anchor_genres` — the full set of genres the user has signalled interest in
- `banned_genres` — genres from `TONE_SHIFTING_GENRES` that are *absent* from the anchor set

`TONE_SHIFTING_GENRES` covers: Action, Fantasy, Horror, Science Fiction, Thriller, Action & Adventure, Sci-Fi & Fantasy, Animation, Documentary, Kids, Reality, and Talk.

**Effect:** If the user rates only comedies, Animation and Sci-Fi are auto-banned. If the user rates an action comedy, Action is *removed* from the blacklist for that session.

**Single dominant genre for `/discover`:** Rather than ORing all anchor genre IDs (which lets The Godfather in via a shared Drama tag), the engine finds the *single most-frequent genre* across the highly-rated anchors and passes only that ID to `with_genres` (AND logic). For a Friends query, this resolves to Comedy (35) only — ensuring every `/discover` result is a comedy first.

**Strict media-type separation:** TV-anchor items only query `/discover/tv`; movie-anchor items only query `/discover/movie`. No cross-medium pollution.

### 4. Reciprocal Rank Fusion (RRF)

After `filter_candidates` gates both lists, RRF merges them into a single ranked output:

```
score(item) = Σ  1 / (k + rank_in_list_i)   for each list i where item appears
```

With `k = 60` (the standard constant that dampens the influence of top ranks):
- Items appearing in *both* the ChromaDB and TMDB lists accumulate contributions from both ranks and naturally float to the top
- Items unique to one list are still represented, preventing either signal from being silently discarded
- No score normalisation or manual weight tuning required

### 5. Rocchio Relevance Feedback

After the user rates Phase 1 candidates, `build_feedback_vector()` computes a shifted query vector:

```
V_new = α · V_query  +  β · Σ(wᵢ · V_liked_i) / |pos|  −  γ · mean(V_disliked)
```

| Parameter | Value | Role |
|---|---|---|
| α (alpha) | 1.0 | Anchors the shift to the original query intent |
| β (beta) | 0.8 | Weight toward positively-rated semantic concepts |
| γ (gamma) | 0.3 | Conservative push away from disliked concepts |
| wᵢ | (rating − 3) / 2 | Graduated weight: 5★ pulls at full β, 4★ at half β |

Ratings of 3 are neutral and ignored. `V_new` is L2-normalised so it remains compatible with ChromaDB's cosine-distance metric. Phase 1 IDs are excluded from the Phase 2 candidate pool, guaranteeing fresh results every time.

---

## Features

| Feature | Details |
|---|---|
| **Semantic Search** | Free-text queries (EN + HE) mapped to 384-dim embeddings — no keyword matching |
| **Rocchio Feedback** | Query vector shifts based on star ratings, not re-filtering |
| **TMDB Discover Integration** | Surfaces popular cultural peers that embedding similarity alone misses |
| **Dynamic Genre Blacklist** | Tone-shifting genres banned per-session based on the user's own taste profile |
| **Reciprocal Rank Fusion** | Smooth rank-level blending of semantic and popularity signals |
| **Strict Media-Type Separation** | TV queries stay TV; movie queries stay movies — no cross-medium noise |
| **Bilingual UI (EN / HE)** | Full Hebrew RTL layout with native-language titles and overviews |
| **Streaming Provider Filter** | Real-time filter by Netflix, Disney+, Apple TV+ etc. via TMDB watch providers |
| **YouTube Trailers** | Official trailers embedded inline via a Streamlit dialog |
| **Safe Search Toggle** | Filters out R / NC-17 / TV-MA certified content |
| **Smart Deduplication** | Sequels swapped for originals; word-overlap collapsing for diverse results |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI & App State | Python 3.11 · Streamlit |
| Vector Database | ChromaDB (Local SQLite + HNSW index, cosine similarity) |
| Embeddings | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual) |
| External Data | TMDB API v3 — metadata, watch providers, trailers, `/discover` |
| Vector Math | NumPy |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` (parallel TMDB enrichment) |
| Config | `python-dotenv` |

---

## Project Structure

```
smart-content-recommender/
├── src/
│   ├── app.py                      # Streamlit entry point — 3-stage reactive UI
│   ├── search_engine.py            # Semantic search, hybrid engine, RRF, genre filtering
│   ├── relevance_feedback.py       # Rocchio algorithm implementation
│   ├── embeddings.py               # Embedding generation + ChromaDB build script
│   ├── tmdb_client.py              # TMDB REST API client (metadata, providers, /discover)
│   └── translations.py             # EN / HE UI strings
├── tests/
│   └── test_relevance_feedback.py  # Unit tests for Rocchio correctness
├── data/
│   ├── movies_cache.json           # Raw TMDB metadata (gitignored)
│   └── chroma_db/                  # Persistent ChromaDB vector index (gitignored)
├── .env                            # API keys — never committed
├── requirements.txt
└── CLAUDE.md
```

---

## Installation & Local Setup

### Prerequisites

- Python 3.11+
- A free [TMDB API](https://www.themoviedb.org/settings/api) account (takes ~2 minutes to create)

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
> The token starts with `eyJ…` and is used as a Bearer token — it is not the same as the shorter v3 API Key.

### 5. Build the vector database

Fetches ~2 000 titles from TMDB, generates embeddings, and writes the ChromaDB index to `data/chroma_db/`. Run once; takes roughly 5–10 minutes depending on connection speed and CPU.

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

**Search query:** *"a funny show about a group of friends in New York"*

1. The engine encodes the query and surfaces 24 candidates ranked by cosine similarity — Friends, Seinfeld, The Office, and similar titles appear at the top.
2. You rate Friends and Seinfeld 5 stars; you rate a drama that slipped in 1 star.
3. **Rocchio shifts** the query vector deeper into comedy-sitcom semantic space and away from drama.
4. **TMDB `/discover`** is called with `with_genres=35` (Comedy, AND logic), `without_genres=27,878,16,…` (Horror, Sci-Fi, Animation, etc. — banned because absent from your anchor), and `vote_count.gte=1500` — HIMYM, New Girl, and Brooklyn Nine-Nine surface immediately.
5. **RRF merges** the ChromaDB semantic results and the TMDB discover results. Titles confirmed by both signals (e.g. a show that ranks high semantically *and* appears in the discover list) float to the top.
6. Filter the final list by whichever streaming service you subscribe to.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers Rocchio correctness using low-dimensional mock vectors — no sentence-transformer model is required, so tests run fast and offline.

---

## License

[MIT](LICENSE)

---

> "This product uses the TMDB API but is not endorsed or certified by TMDB."
