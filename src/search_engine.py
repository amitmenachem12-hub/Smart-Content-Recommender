import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_EMBEDDINGS_PATH = os.path.join(_DATA_DIR, "movies_with_embeddings.json")
_MODEL_NAME = "all-MiniLM-L6-v2"

_STOP_WORDS = {"the", "a", "an", "of", "in", "at", "on", "and", "or", "to", "is", "it"}
_MIN_SCORE = 0.25
_GENRE_BOOST = 0.15

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


def swap_sequel_for_original(selected_movie: dict, all_movies_data: list[dict]) -> dict:
    match = _SEQUEL_RE.match(selected_movie["title"])
    if not match:
        return selected_movie
    base_title = match.group(1).strip()
    for movie in all_movies_data:
        if movie["title"] == base_title:
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


def load_movies() -> list[dict]:
    with open(_EMBEDDINGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def semantic_search(
    query_text: str,
    top_k: int = 5,
    pool_size: int = 30,
    model: SentenceTransformer | None = None,
    movies_data: list[dict] | None = None,
) -> dict:
    movies: list[dict] = movies_data if movies_data is not None else load_movies()

    target_media: str | None = None
    query_lower = query_text.lower()
    if re.search(r"\b(show|series|tv)\b", query_lower):
        target_media = "TV Show"
    elif re.search(r"\b(movie|film)\b", query_lower):
        target_media = "Movie"

    if target_media is not None:
        movies = [m for m in movies if m.get("media_type") == target_media]

    if model is None:
        model = load_model()
    query_vec = model.encode(expand_query(query_text))

    emb_matrix = np.array([m["embedding"] for m in movies])

    # Normalise once and use dot product as cosine similarity.
    query_norm = query_vec / np.linalg.norm(query_vec)
    emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / np.where(emb_norms == 0, 1, emb_norms)

    scores = emb_normed @ query_norm

    _NEGATIVE_KEYWORDS = [
        "eerie", "trouble", "plagued", "murder", "blood",
        "haunted", "dark", "horror", "death", "killer",
    ]
    _LIGHTHEARTED_TRIGGERS = ("funny", "comedy", "laugh", "relax")

    if any(w in query_text.lower() for w in _LIGHTHEARTED_TRIGGERS):
        for idx, movie in enumerate(movies):
            overview_lower = movie.get("overview", "").lower()
            if any(kw in overview_lower for kw in _NEGATIVE_KEYWORDS):
                scores[idx] -= 0.15

    target_genres: set[str] = set()
    for keyword, genres in _GENRE_BOOST_MAP.items():
        if keyword in query_lower:
            target_genres.update(genres)

    if target_genres:
        for idx, movie in enumerate(movies):
            movie_genres = {
                (g["name"] if isinstance(g, dict) else g)
                for g in movie.get("genres", [])
            }
            if movie_genres & target_genres:
                scores[idx] += _GENRE_BOOST

    sorted_indices = np.argsort(scores)[::-1]

    initial_pool: list[dict] = []
    selected_title_words: list[set[str]] = []

    for i in sorted_indices:
        if len(initial_pool) >= pool_size:
            break
        if scores[i] < _MIN_SCORE:
            break
        movie = swap_sequel_for_original(movies[i], movies)
        candidate_words = _title_words(movie["title"])
        if _is_too_similar(candidate_words, selected_title_words):
            continue

        initial_pool.append({**movie, "score": float(scores[i])})
        selected_title_words.append(candidate_words)

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
