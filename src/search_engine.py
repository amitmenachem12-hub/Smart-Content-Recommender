import json
import os
import re

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")
_COLLECTION_NAME = "movies"
_MODEL_NAME = "all-MiniLM-L6-v2"

_STOP_WORDS = {"the", "a", "an", "of", "in", "at", "on", "and", "or", "to", "is", "it"}
_MIN_SCORE = 0.25
_MATURE_RATINGS = frozenset(["R", "NC-17", "X", "TV-MA"])

# Maps query keywords → TMDB genre names (covers both Movie and TV variants).
_GENRE_BOOST_MAP: dict[str, list[str]] = {
    "comedy":      ["Comedy"],
    "funny":       ["Comedy"],
    "sitcom":      ["Comedy"],
    "laugh":       ["Comedy"],
    "action":      ["Action", "Action & Adventure"],
    "sci-fi":      ["Sci-Fi & Fantasy", "Science Fiction"],
    "scifi":       ["Sci-Fi & Fantasy", "Science Fiction"],
    "space":       ["Sci-Fi & Fantasy", "Science Fiction"],
    "horror":      ["Horror"],
    "scary":       ["Horror"],
    "drama":       ["Drama"],
    "romance":     ["Romance"],
    "romantic":    ["Romance"],
    "thriller":    ["Thriller"],
    "crime":       ["Crime"],
    "mystery":     ["Mystery"],
    "fantasy":     ["Fantasy", "Sci-Fi & Fantasy"],
    "adventure":   ["Adventure", "Action & Adventure"],
    "animation":   ["Animation"],
    "documentary": ["Documentary"],
    "family":      ["Family"],
}

# Matches "Part II/2/Three", "Vol. 2", standalone digits, or 2+ Roman numeral chars at the end.
_SEQUEL_RE = re.compile(
    r"^(.+?)\s+(?:Part\s+\w+|Vol\.?\s+\w+|\d+|[IVX]{2,})$",
    re.IGNORECASE,
)

_NEGATIVE_KEYWORDS = frozenset([
    "eerie", "trouble", "plagued", "murder", "blood",
    "haunted", "dark", "horror", "death", "killer",
])
_LIGHTHEARTED_TRIGGERS = frozenset(["funny", "comedy", "laugh", "relax"])


def _title_words(title: str) -> set[str]:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return {w for w in words if w not in _STOP_WORDS}


def _is_too_similar(candidate: set[str], selected: list[set[str]], threshold: float = 0.5) -> bool:
    for seen in selected:
        smaller = min(len(candidate), len(seen))
        if smaller == 0:
            continue
        if len(candidate & seen) / smaller > threshold:
            return True
    return False


def _metadata_to_movie(doc_id: str, metadata: dict) -> dict:
    return {
        "id": doc_id,
        "title": metadata["title"],
        "overview": metadata["overview"],
        "media_type": metadata["media_type"],
        "genres": json.loads(metadata["genres_json"]),
        "vote_average": metadata["vote_average"],
        "poster_path": metadata["poster_path"],
        "watch_providers": json.loads(metadata["watch_providers_json"]),
        "certification": metadata.get("certification", "Unrated"),
    }


def swap_sequel_for_original(selected_movie: dict, collection: chromadb.Collection) -> dict:
    match = _SEQUEL_RE.match(selected_movie["title"])
    if not match:
        return selected_movie
    base_title = match.group(1).strip()
    result = collection.get(
        where={"title": {"$eq": base_title}},
        include=["metadatas", "embeddings"],
        limit=1,
    )
    if result["ids"]:
        movie = _metadata_to_movie(result["ids"][0], result["metadatas"][0])
        movie["embedding"] = result["embeddings"][0]
        return movie
    return selected_movie


def expand_query(query_text: str) -> str:
    lower = query_text.lower()
    anchors: list[str] = []
    if any(w in lower for w in ("funny", "comedy", "laugh")):
        anchors.append("lighthearted, uplifting, romantic comedy, sitcom, fun, feel-good")
    if any(w in lower for w in ("tired", "evening", "exhausted")):
        anchors.append("relaxing, easy to watch, soothing")
    if not anchors:
        return query_text
    return query_text + " " + " ".join(anchors)


def load_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return client.get_collection(name=_COLLECTION_NAME)


def semantic_search(
    query_text: str,
    top_k: int = 5,
    pool_size: int = 30,
    model: SentenceTransformer | None = None,
    collection: chromadb.Collection | None = None,
    safe_search: bool = False,
) -> dict:
    if collection is None:
        collection = load_collection()

    query_lower = query_text.lower()

    # --- Media type filter ---
    target_media: str | None = None
    if re.search(r"\b(show|series|tv)\b", query_lower):
        target_media = "TV Show"
    elif re.search(r"\b(movie|film)\b", query_lower):
        target_media = "Movie"

    # --- Explicit genre intent (word-boundary matching avoids false positives) ---
    target_genres: set[str] = set()
    for keyword, genres in _GENRE_BOOST_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
            target_genres.update(genres)

    # --- Encode query ---
    if model is None:
        model = load_model()
    query_vec = model.encode(expand_query(query_text))
    query_norm = query_vec / np.linalg.norm(query_vec)

    # --- Query ChromaDB (no where filter) ---
    # ChromaDB's $contains operator is unreliable for list-valued fields stored
    # as strings. Fetch a large unfiltered candidate set and apply genre/media
    # constraints in Python where the logic is transparent and testable.
    fetch_n = min(pool_size * 4, collection.count())
    if fetch_n == 0:
        return {"initial_pool": [], "top_k_results": [], "query_vector": query_norm}

    results = collection.query(
        query_embeddings=[query_norm.tolist()],
        n_results=fetch_n,
        include=["embeddings", "metadatas", "distances"],
    )

    # ChromaDB returns cosine *distance*; convert to similarity (1 − distance).
    candidates: list[tuple[dict, float]] = [
        ({**_metadata_to_movie(doc_id, meta), "embedding": emb}, 1.0 - dist)
        for doc_id, dist, meta, emb in zip(
            results["ids"][0], results["distances"][0],
            results["metadatas"][0], results["embeddings"][0],
        )
    ]

    # --- Python-side hard filtering ---
    # Apply media-type, genre, and safe-search constraints here, not in
    # ChromaDB, so the logic is not dependent on ChromaDB operator support.
    if target_media is not None or target_genres or safe_search:
        filtered: list[tuple[dict, float]] = []
        for movie, score in candidates:
            if target_media is not None and movie["media_type"] != target_media:
                continue
            pre_genre_names = {g["name"] if isinstance(g, dict) else g for g in movie["genres"]}
            if target_genres and not (target_genres & pre_genre_names):
                continue
            if safe_search and movie.get("certification", "Unrated") in _MATURE_RATINGS:
                continue
            filtered.append((movie, score))
        candidates = filtered

    # --- Lighthearted penalty (post-retrieval, re-sort only when triggered) ---
    if any(w in query_lower for w in _LIGHTHEARTED_TRIGGERS):
        candidates = [
            (movie, score - 0.15)
            if any(kw in movie["overview"].lower() for kw in _NEGATIVE_KEYWORDS)
            else (movie, score)
            for movie, score in candidates
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

    # --- Build pool with title dedup ---
    initial_pool: list[dict] = []
    selected_title_words: list[set[str]] = []

    for movie, score in candidates:
        if len(initial_pool) >= pool_size:
            break
        if score < _MIN_SCORE:
            break
        movie = swap_sequel_for_original(movie, collection)
        # swap_sequel_for_original returns a raw ChromaDB record that was never
        # run through the pre-filter above, so both hard constraints must be
        # checked again here before the item can enter the pool.
        if target_media is not None and movie["media_type"] != target_media:
            continue
        movie_genre_names = {g["name"] if isinstance(g, dict) else g for g in movie["genres"]}
        if target_genres and not (target_genres & movie_genre_names):
            continue
        if safe_search and movie.get("certification", "Unrated") in _MATURE_RATINGS:
            continue
        candidate_words = _title_words(movie["title"])
        if _is_too_similar(candidate_words, selected_title_words):
            continue
        initial_pool.append({**movie, "score": score})
        selected_title_words.append(candidate_words)

    # --- Attach justification for top-k results ---
    top_k_results: list[dict] = []
    for candidate in initial_pool[:top_k]:
        genres = candidate.get("genres", [])
        genre_names = [g["name"] if isinstance(g, dict) else g for g in genres]
        justification = "Recommended because it strongly matches your intent" + (
            f" and features genres: {genre_names}" if genre_names else ""
        )
        top_k_results.append({
            **{k: v for k, v in candidate.items() if k != "embedding"},
            "justification": justification,
        })

    return {"initial_pool": initial_pool, "top_k_results": top_k_results, "query_vector": query_norm}


if __name__ == "__main__":
    query = "A funny show about a group of friends in New York that one of the charecthers fell in love"
    output = semantic_search(query, top_k=5)
    print(f"Query: {query}\n")
    top_k_results = output["top_k_results"]
    if not top_k_results:
        print("No highly relevant matches found for your exact request.")
    else:
        for rank, movie in enumerate(top_k_results, start=1):
            print(f"{rank}. {movie['title']} [{movie.get('media_type', 'Unknown')}] ({movie['score']:.4f})")
            print(f"   {movie['overview']}")
            print(f"   {movie['justification']}\n")
