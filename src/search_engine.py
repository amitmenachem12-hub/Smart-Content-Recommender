import json
import os
import re
from collections.abc import Callable

import streamlit as st

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")
_COLLECTION_NAME = "movies"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_STOP_WORDS = {"the", "a", "an", "of", "in", "at", "on", "and", "or", "to", "is", "it"}
_MIN_SCORE = 0.25
# Score floor for TMDB-validated recs so they rank competitively against vector results.
_TMDB_REC_MIN_SCORE = 0.30
# Score bonus for items confirmed by BOTH the TMDB signal AND vector search.
_TMDB_BOOST = 0.12
_MATURE_RATINGS = frozenset(["R", "NC-17", "X", "TV-MA"])

# Genres that are culturally incompatible with a given anchor primary genre.
# A TMDB rec is rejected if its genre list intersects the applicable blocked set.
# Safe secondaries (Drama, Romance, Family, Crime) are deliberately absent so
# shows like HIMYM [Comedy, Drama, Romance] and New Girl [Comedy, Drama] pass.
_GENRE_BLOCKLIST: dict[str, frozenset[str]] = {
    "Comedy": frozenset({
        "Sci-Fi & Fantasy", "Science Fiction",
        "Mystery", "Horror",
        "Action & Adventure", "Action",
        "War & Politics", "War",
    }),
}

# TMDB genre IDs considered tone-shifting or non-narrative.
# Banned from discover results unless the anchor itself carries the genre.
TONE_SHIFTING_GENRES = {
    28,     # Action
    14,     # Fantasy
    27,     # Horror
    878,    # Science Fiction
    53,     # Thriller
    10759,  # Action & Adventure (TV)
    10765,  # Sci-Fi & Fantasy (TV)
    16,     # Animation
    99,     # Documentary
    10762,  # Kids (TV)
    10764,  # Reality (TV)
    10767,  # Talk (TV)
}

# Name-string equivalents of TONE_SHIFTING_GENRES for filtering against the
# genre name strings stored in ChromaDB metadata.
_TONE_SHIFTING_GENRE_NAMES: frozenset[str] = frozenset({
    "Action", "Fantasy", "Horror", "Science Fiction",
    "Thriller", "Action & Adventure", "Sci-Fi & Fantasy",
    "Animation", "Documentary", "Kids", "Reality", "Talk",
})

# Maps genre name strings (as stored in ChromaDB) → numeric TMDB genre IDs.
# Used to convert build_genre_profile output to IDs for the /discover endpoint.
_GENRE_NAME_TO_ID: dict[str, int] = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
    # TV-specific variants
    "Action & Adventure": 10759,
    "Kids": 10762,
    "News": 10763,
    "Reality": 10764,
    "Sci-Fi & Fantasy": 10765,
    "Soap": 10766,
    "Talk": 10767,
    "War & Politics": 10768,
}

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

# Maps frozensets of trigger keywords → (ISO 639-1 language code, ISO 3166-1 country codes).
# A result passes if its original_language matches OR any of its origin_country codes match.
_COUNTRY_FILTER_MAP: list[tuple[frozenset[str], str, list[str]]] = [
    (frozenset(["israeli", "israel"]), "he", ["IL"]),
]

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
        "original_language": metadata.get("original_language", ""),
        "origin_country": json.loads(metadata.get("origin_country_json", "[]")),
    }


def _detect_country_filter(query_lower: str) -> tuple[str, list[str]] | None:
    """Returns (language_code, country_codes) if a nationality keyword matches, else None."""
    for keywords, lang, countries in _COUNTRY_FILTER_MAP:
        if any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in keywords):
            return (lang, countries)
    return None


def _passes_country_filter(movie: dict, lang: str, countries: list[str]) -> bool:
    return (
        movie.get("original_language") == lang
        or any(c in movie.get("origin_country", []) for c in countries)
    )


def _build_hard_filter(
    query_text: str, safe_search: bool,
) -> tuple[str | None, Callable[[dict], bool]]:
    """Derive the strict media/genre/country/safety constraints implied by the query.

    Returns the target media type (for ChromaDB's $eq pre-filter) and a predicate
    that the Python side applies to every candidate.
    """
    query_lower = query_text.lower()

    target_media: str | None = None
    if re.search(r"\b(show|series|tv)\b", query_lower):
        target_media = "TV Show"
    elif re.search(r"\b(movie|film)\b", query_lower):
        target_media = "Movie"

    # Word-boundary matching avoids false positives.
    target_genres: set[str] = set()
    for keyword, genres in _GENRE_BOOST_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
            target_genres.update(genres)

    country_filter = _detect_country_filter(query_lower)

    def passes(movie: dict) -> bool:
        if target_media is not None and movie["media_type"] != target_media:
            return False
        if target_genres:
            names = {g["name"] if isinstance(g, dict) else g for g in movie["genres"]}
            if not (target_genres & names):
                return False
        if country_filter is not None and not _passes_country_filter(movie, *country_filter):
            return False
        if safe_search and movie.get("certification", "Unrated") in _MATURE_RATINGS:
            return False
        return True

    return target_media, passes


def _with_justification(movie: dict) -> dict:
    genre_names = [g["name"] if isinstance(g, dict) else g for g in movie.get("genres", [])]
    justification = "Recommended because it strongly matches your intent" + (
        f" and features genres: {genre_names}" if genre_names else ""
    )
    return {
        **{k: v for k, v in movie.items() if k != "embedding"},
        "justification": justification,
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
    target_media, passes_hard_filter = _build_hard_filter(query_text, safe_search)

    # --- Encode query ---
    if model is None:
        model = load_model()
    query_vec = model.encode(expand_query(query_text))
    query_norm = query_vec / np.linalg.norm(query_vec)

    # --- Query ChromaDB ---
    # media_type is a flat string field — $eq filtering is reliable and applied
    # here to enforce strict media-type intent at the database level.
    # Genre/country filtering remains Python-side because those fields are stored
    # as pipe-delimited strings where ChromaDB's $contains is unreliable.
    fetch_n = min(pool_size * 5, collection.count())
    if fetch_n == 0:
        return {"initial_pool": [], "top_k_results": [], "query_vector": query_norm}

    where_clause = {"media_type": {"$eq": target_media}} if target_media else None
    results = collection.query(
        query_embeddings=[query_norm.tolist()],
        n_results=fetch_n,
        include=["embeddings", "metadatas", "distances"],
        **({"where": where_clause} if where_clause else {}),
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
    # Apply media-type, genre, country/language, and safe-search constraints here,
    # not in ChromaDB, so the logic is not dependent on ChromaDB operator support.
    candidates = [(movie, score) for movie, score in candidates if passes_hard_filter(movie)]

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
        if not passes_hard_filter(movie):
            continue
        candidate_words = _title_words(movie["title"])
        if _is_too_similar(candidate_words, selected_title_words):
            continue
        initial_pool.append({**movie, "score": score})
        selected_title_words.append(candidate_words)

    top_k_results = [_with_justification(candidate) for candidate in initial_pool[:top_k]]

    return {"initial_pool": initial_pool, "top_k_results": top_k_results, "query_vector": query_norm}


def _validate_tmdb_recs(
    composite_ids: list[str],
    collection: chromadb.Collection,
    exclude_ids: set[str],
    passes_hard_filter: Callable[[dict], bool],
    anchor_primary_genres: set[str],
    query_vector: np.ndarray,
) -> dict[str, dict]:
    """Validate TMDB-sourced candidates and return those that pass the blocklist veto.

    Returns a mapping of composite_id → item_dict (with "score" set to cosine
    similarity, floored at _TMDB_REC_MIN_SCORE).  Title deduplication is handled
    by the caller.  IDs absent from the ChromaDB index are silently skipped.

    Targeted Blocklist Veto:
      Unions the _GENRE_BLOCKLIST entries for every anchor primary genre to build
      a set of blocked genres.  A rec is rejected only if its genre list intersects
      that blocked set — safe secondaries (Drama, Romance, Family) pass freely.

      Example with a Comedy anchor:
        "Wednesday" [Comedy, Mystery, Sci-Fi & Fantasy] → Mystery & Sci-Fi blocked → REJECTED
        "HIMYM"     [Comedy, Drama, Romance]            → no blocked genres        → PASS
        "New Girl"  [Comedy, Drama]                     → no blocked genres        → PASS

      When no anchor primary genre maps to a _GENRE_BLOCKLIST entry (or no items
      are rated ≥ 4), the blocklist is empty and a vector similarity floor of
      _TMDB_REC_MIN_SCORE is applied as a lightweight fallback quality gate.
    """
    candidates = [id_ for id_ in composite_ids if id_ not in exclude_ids]
    if not candidates:
        return {}
    result = collection.get(ids=candidates, include=["metadatas", "embeddings"])
    q_vec = np.asarray(query_vector, dtype=np.float32)

    # Build the applicable blocked-genre set from all anchor primary genres.
    blocked_genres: set[str] = set()
    for g in anchor_primary_genres:
        blocked_genres.update(_GENRE_BLOCKLIST.get(g, frozenset()))

    validated: dict[str, dict] = {}
    for doc_id, meta, emb_raw in zip(result["ids"], result["metadatas"], result["embeddings"]):
        if doc_id in validated:
            continue
        movie = _metadata_to_movie(doc_id, meta)
        if not passes_hard_filter(movie):
            continue

        rec_genre_set = set(movie.get("genres", []))

        # Blocklist veto: reject if the rec carries any blocked genre.
        if blocked_genres:
            if rec_genre_set & blocked_genres:
                continue

        # Compute cosine similarity for scoring.
        emb = np.asarray(emb_raw, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        similarity = float(emb @ q_vec)

        # Fallback quality gate when no anchor has a blocklist entry.
        if not blocked_genres and similarity < _TMDB_REC_MIN_SCORE:
            continue

        score = max(similarity, _TMDB_REC_MIN_SCORE)
        item = {**_with_justification(movie), "score": score}
        item["justification"] = "Fans of your top picks also loved this"
        validated[doc_id] = item
    return validated


def search_by_vector(
    query_vector: np.ndarray,
    query_text: str,
    n_results: int,
    exclude_items: list[dict],
    collection: chromadb.Collection | None = None,
    safe_search: bool = False,
    tmdb_rec_ids: list[str] | None = None,
    anchor_primary_genres: set[str] | None = None,
) -> list[dict]:
    """Phase 2 hybrid retrieval: true union of TMDB recs and Rocchio vector search.

    Steps:
      1. Validate TMDB candidates via the targeted blocklist veto → valid_tmdb_map.
      2. Run vector search (ChromaDB, Rocchio-shifted), oversample to absorb losses.
      3. Process vector hits; apply _TMDB_BOOST to any hit also in valid_tmdb_map
         (double-signal items float to the top after the final sort).
      4. Inject validated TMDB items not surfaced by vector search.
      5. Sort the blended pool by final score, trim to n_results.

    The blocklist veto allows safe secondary genres (Drama, Romance) through while
    rejecting culturally incompatible ones (Sci-Fi, Mystery, Horror).
    Items confirmed by both signals rank highest.  The streaming-provider filter
    applied by the caller operates on this sorted list.
    """
    if collection is None:
        collection = load_collection()

    target_media, passes_hard_filter = _build_hard_filter(query_text, safe_search)
    exclude_ids = {item["id"] for item in exclude_items}
    selected_title_words = [_title_words(item["title"]) for item in exclude_items]

    # Step 1: Validate TMDB candidates via the targeted blocklist veto.
    valid_tmdb_map: dict[str, dict] = {}
    if tmdb_rec_ids:
        valid_tmdb_map = _validate_tmdb_recs(
            tmdb_rec_ids, collection, exclude_ids, passes_hard_filter,
            anchor_primary_genres or set(), query_vector,
        )

    # Step 2: Vector search — oversample; Python-side filters discard many hits.
    fetch_n = min((n_results + len(exclude_items)) * 5, collection.count())
    blended: list[dict] = []
    seen_ids: set[str] = set()

    if fetch_n > 0:
        where_clause = {"media_type": {"$eq": target_media}} if target_media else None
        results = collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=fetch_n,
            include=["metadatas", "distances"],
            **({"where": where_clause} if where_clause else {}),
        )

        # Step 3: Collect vector hits; boost those also validated by TMDB.
        for doc_id, dist, meta in zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0],
        ):
            if len(blended) >= n_results:
                break
            if doc_id in exclude_ids:
                continue
            movie = _metadata_to_movie(doc_id, meta)
            if not passes_hard_filter(movie):
                continue
            words = _title_words(movie["title"])
            if _is_too_similar(words, selected_title_words):
                continue
            rec = {**_with_justification(movie), "score": 1.0 - dist}
            if doc_id in valid_tmdb_map:
                rec["score"] += _TMDB_BOOST
                rec["justification"] = "Matched by both semantic and collaborative signals"
            blended.append(rec)
            seen_ids.add(doc_id)
            selected_title_words.append(words)

    # Step 4: Inject validated TMDB items not surfaced by vector search.
    for tid, trec in valid_tmdb_map.items():
        if tid in seen_ids or tid in exclude_ids:
            continue
        words = _title_words(trec["title"])
        if _is_too_similar(words, selected_title_words):
            continue
        blended.append(trec)
        selected_title_words.append(words)

    # Step 5: Sort by final score, trim to n_results.
    blended.sort(key=lambda x: x["score"], reverse=True)
    return blended[:n_results]


def build_genre_profile(anchor_items: list[dict]) -> tuple[set[str], set[str]]:
    """Derive anchor genre set and the tone-shifting genres absent from it.

    Returns:
        anchor_genres  — all genre name strings from anchor_items.
        banned_genres  — tone-shifting genre names NOT present in the anchor set;
                         items carrying any of these should be excluded.
    """
    anchor_genres: set[str] = set()
    for item in anchor_items:
        for g in item.get("genres", []):
            anchor_genres.add(g["name"] if isinstance(g, dict) else g)
    banned_genres = _TONE_SHIFTING_GENRE_NAMES - anchor_genres
    return anchor_genres, banned_genres


def filter_candidates(
    candidates: list[dict],
    anchor_genres: set[str],
    banned_genres: set[str],
    seen_ids: set[str],
) -> list[dict]:
    """Filter a candidate list against genre profile constraints.

    Mutates seen_ids in-place so duplicate items across calls are suppressed.
    Skips items with no genre overlap with the anchor, and items that carry
    any banned (tone-shifting) genre.
    """
    result: list[dict] = []
    for item in candidates:
        item_id = item.get("id", "")
        if item_id in seen_ids:
            continue
        item_genres = {g["name"] if isinstance(g, dict) else g for g in item.get("genres", [])}
        if not (item_genres & anchor_genres):
            continue
        if item_genres & banned_genres:
            continue
        seen_ids.add(item_id)
        result.append(item)
    return result


def reciprocal_rank_fusion(
    list1: list[dict], list2: list[dict], k: int = 60
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Each item scores 1/(k + rank) per list it appears in; scores are summed
    across lists.  Returns original item dicts sorted by descending RRF score.
    Items appearing in both lists receive contributions from both ranks.
    """
    scores: dict[str, float] = {}
    items_by_id: dict[str, dict] = {}
    for rank, item in enumerate(list1, start=1):
        item_id = item.get("id", "")
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        items_by_id[item_id] = item
    for rank, item in enumerate(list2, start=1):
        item_id = item.get("id", "")
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        items_by_id[item_id] = item
    return [items_by_id[id_] for id_ in sorted(scores, key=scores.__getitem__, reverse=True)]


def _discover_via_genre_profile(
    anchor_items: list[dict],
    banned_genres: set[str],
    collection: chromadb.Collection,
    phase_1_seen_ids: set[str],
    max_per_type: int = 40,
) -> list[dict]:
    """Fetch popular TMDB items via /discover using the single dominant anchor genre.

    Groups anchor items by media type (TV Show / Movie) and finds the most
    frequently appearing genre within each group.  Only THAT ONE genre ID is
    passed to with_genres (AND logic), so /discover stays strictly on topic —
    e.g. Comedy=35 only, not Comedy|Drama which would surface unrelated films.
    The endpoint is only called for media types actually present in the anchors,
    preventing cross-medium pollution (no movies returned for a TV-show query).
    """
    from tmdb_client import fetch_discover_by_genres

    banned_ids = [_GENRE_NAME_TO_ID[g] for g in banned_genres if g in _GENRE_NAME_TO_ID]

    def _dominant_genre_id(items: list[dict]) -> int | None:
        counts: dict[str, int] = {}
        for item in items:
            for g in item.get("genres", []):
                name = g["name"] if isinstance(g, dict) else g
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        return _GENRE_NAME_TO_ID.get(max(counts, key=counts.__getitem__))

    groups = [
        ([it for it in anchor_items if it.get("media_type") == "TV Show"], "TV Show", "TV Show"),
        ([it for it in anchor_items if it.get("media_type") == "Movie"],   "Movie",   "Movie"),
    ]

    composite_ids: list[str] = []
    seen: set[str] = set(phase_1_seen_ids)

    for group_items, media_type, prefix in groups:
        if not group_items:
            continue
        dominant_id = _dominant_genre_id(group_items)
        if dominant_id is None:
            continue
        raw_ids = fetch_discover_by_genres(
            [dominant_id], banned_ids, media_type=media_type, max_results=max_per_type,
        )
        for rid in raw_ids:
            cid = f"{prefix}_{rid}"
            if cid not in seen:
                composite_ids.append(cid)
                seen.add(cid)

    if not composite_ids:
        return []

    result = collection.get(ids=composite_ids, include=["metadatas"])
    return [
        _metadata_to_movie(doc_id, meta)
        for doc_id, meta in zip(result["ids"], result["metadatas"])
    ]


def get_hybrid_recommendations(
    chroma_results: list[dict],
    anchor_items: list[dict],
    phase_1_seen_ids: set[str],
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    """Discover-based hybrid Phase 2 engine.

    Fetches widely-popular TMDB titles via /discover filtered by the anchor
    genre profile, then fuses them with ChromaDB vector results via RRF.
    Falls back to ChromaDB-only results when no items are rated ≥ 4
    (empty anchor_genres → no discover fetch possible).
    """
    if collection is None:
        collection = load_collection()

    anchor_genres, banned_genres = build_genre_profile(anchor_items)

    if not anchor_genres:
        with st.sidebar.expander("Debug: Raw API & Chroma Results", expanded=True):
            st.write("Raw TMDB Candidates:", [])
            st.write("Raw Chroma Candidates:", [x.get("title") for x in chroma_results])
        return [_with_justification(item) for item in chroma_results]

    tmdb_results = _discover_via_genre_profile(
        anchor_items, banned_genres, collection, phase_1_seen_ids
    )

    with st.sidebar.expander("Debug: Raw API & Chroma Results", expanded=True):
        st.write("Raw TMDB Candidates:", [x.get("title") for x in tmdb_results])
        st.write("Raw Chroma Candidates:", [x.get("title") for x in chroma_results])

    chroma_seen: set[str] = set(phase_1_seen_ids)
    tmdb_seen: set[str] = set(phase_1_seen_ids)
    filtered_chroma = filter_candidates(chroma_results, anchor_genres, banned_genres, chroma_seen)
    filtered_tmdb = filter_candidates(tmdb_results, anchor_genres, banned_genres, tmdb_seen)
    fused = reciprocal_rank_fusion(filtered_chroma, filtered_tmdb)
    return [_with_justification(item) for item in fused]


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
