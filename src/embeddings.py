import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "movies_cache.json")
_OUTPUT_PATH = os.path.join(_DATA_DIR, "movies_with_embeddings.json")
_CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")
_COLLECTION_NAME = "movies"
_MODEL_NAME = "all-MiniLM-L6-v2"
_UPSERT_BATCH = 500


def generate_movie_embeddings() -> None:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        movies: list[dict] = json.load(f)

    model = SentenceTransformer(_MODEL_NAME)

    texts = [
        (
            f"Type: {m.get('media_type') or ''}. "
            f"Genres: {', '.join(m.get('genres') or [])}. "
            f"Country: {', '.join(m.get('origin_country') or [])}. "
            f"Language: {m.get('original_language') or ''}. "
            f"Title: {m.get('title') or ''}. "
            f"Overview: {m.get('overview') or ''}"
        )
        for m in movies
    ]
    embeddings = model.encode(texts, show_progress_bar=True)

    movies_with_embeddings = [
        {**movie, "embedding": embedding.tolist()}
        for movie, embedding in zip(movies, embeddings)
    ]

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(movies_with_embeddings, f, indent=4)


def build_chroma_collection() -> None:
    with open(_OUTPUT_PATH, encoding="utf-8") as f:
        movies: list[dict] = json.load(f)

    client = chromadb.PersistentClient(path=_CHROMA_DIR)

    # Always rebuild from scratch so the collection stays in sync with the JSON.
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    for movie in movies:
        genres: list[str] = [
            g["name"] if isinstance(g, dict) else g
            for g in (movie.get("genres") or [])
        ]
        # Pipe-delimited with boundary markers so "|Comedy|" won't partially
        # match a genre whose name contains the same substring.
        genres_str = "|" + "|".join(genres) + "|" if genres else "||"

        # Prefix TMDB id with media type to guarantee uniqueness across the
        # separate movie and TV show ID namespaces that TMDB uses.
        doc_id = f"{movie.get('media_type', 'unknown')}_{movie['id']}"

        ids.append(doc_id)
        embeddings.append(movie["embedding"])
        metadatas.append({
            "title": movie.get("title") or "",
            "overview": movie.get("overview") or "",
            "media_type": movie.get("media_type") or "",
            "genres_str": genres_str,
            "genres_json": json.dumps(genres),
            "vote_average": float(movie.get("vote_average") or 0.0),
            "poster_path": movie.get("poster_path") or "",
            "watch_providers_json": json.dumps(movie.get("watch_providers") or []),
            "certification": movie.get("certification") or "Unrated",
            "original_language": movie.get("original_language") or "",
            "origin_country_json": json.dumps(movie.get("origin_country") or []),
        })
        documents.append(movie.get("title") or "")

    total = len(ids)
    for i in range(0, total, _UPSERT_BATCH):
        collection.upsert(
            ids=ids[i : i + _UPSERT_BATCH],
            embeddings=embeddings[i : i + _UPSERT_BATCH],
            metadatas=metadatas[i : i + _UPSERT_BATCH],
            documents=documents[i : i + _UPSERT_BATCH],
        )
        print(f"  upserted {min(i + _UPSERT_BATCH, total)}/{total}")

    print(
        f"Collection '{_COLLECTION_NAME}' ready — {total} movies"
        f" at {os.path.abspath(_CHROMA_DIR)}"
    )


if __name__ == "__main__":
    generate_movie_embeddings()
    print(f"Embeddings saved to {os.path.abspath(_OUTPUT_PATH)}\n")
    print("Building ChromaDB collection…")
    build_chroma_collection()
