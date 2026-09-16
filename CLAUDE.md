# CLAUDE.md — Smart Content Recommender

## Project purpose

Semantic content recommendation system that reduces choice fatigue by matching user intent to movie/TV content via embeddings, not just metadata filters.

## Architecture (planned)

```
smart-content-recommender/
├── src/
│   ├── api/          # TMDB API client
│   ├── embeddings/   # Embedding generation and similarity search
│   ├── recommender/  # Core recommendation logic
│   └── ui/           # CLI or Streamlit interface
├── tests/
├── .env              # API keys (never committed)
├── requirements.txt
└── CLAUDE.md
```

## Technical constraints

- **Pure Python only.** No React, Next.js, or JavaScript frontend frameworks.
- **UI layer:** CLI (`argparse`) or Streamlit — nothing heavier.
- **External data:** TMDB API only. No scraping or other unauthorized sources.
- **Embeddings:** Use a lightweight sentence-transformer model (e.g., `all-MiniLM-L6-v2`) unless a better fit is identified.
- **No premature abstraction.** Add layers only when the feature demands it.

## Coding conventions

- Python 3.11+.
- Type hints on all public functions.
- No comments that restate what the code does — only explain non-obvious *why*.
- One concern per module; keep files short.
- Tests live in `tests/` and mirror the `src/` structure.
- Use `python-dotenv` to load `.env`; never hard-code keys.

## Environment variables

| Variable | Purpose |
|---|---|
| `TMDB_API_KEY` | TMDB v3 API key |

## Key decisions

- Semantic search over keyword/genre filters to address choice fatigue.
- Embeddings computed at query time initially; caching added only if latency becomes a problem.
- TMDB chosen for its free tier and comprehensive metadata.
