# Smart Content Recommender

A Python-based content recommendation system designed to reduce **choice fatigue** by helping users discover movies and TV shows that genuinely match their tastes — not just what's trending.

## Problem

Streaming platforms surface content through popularity and engagement metrics, which often leaves users overwhelmed by irrelevant options. Scrolling through hundreds of titles without finding something to watch is a form of choice fatigue.

## Goal

Replace surface-level filtering (genre, rating) with **semantic understanding** of user preferences. By embedding natural-language descriptions of content and user intent, the system finds recommendations that are contextually relevant, not just statistically popular.

## Approach

1. Fetch content metadata from the [TMDB API](https://www.themoviedb.org/documentation/api).
2. Generate semantic embeddings for plot summaries and user queries.
3. Rank candidates by similarity to surface the most relevant matches.
4. Present results through a simple CLI or Streamlit interface.

## Stack

- Python 3.11+
- TMDB API (content data)
- Sentence embeddings (semantic search)
- Streamlit or CLI (UI)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env .env.local  # add your TMDB_API_KEY
```
