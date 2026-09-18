import numpy as np


def _build_justification(item: dict) -> str:
    genres = item.get("genres", [])
    genre_names = [g["name"] if isinstance(g, dict) else g for g in genres]
    suffix = f" and features genres: {genre_names}" if genre_names else ""
    return "Recommended because it strongly matches your intent" + suffix


def apply_user_feedback(
    original_query_vector: np.ndarray,
    rated_items: dict[str, int],
    candidate_pool: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Refine results using Rocchio-style relevance feedback.

    Adjusts the original query vector toward highly-rated items and away from
    low-rated ones, then re-ranks the remaining candidates against the updated
    vector.  Title lookup is case-insensitive.

    Rating → weight mapping is linear over [1, 10] → [-1.0, +1.0]:
        weight = (rating - 5.5) / 4.5
    Ratings near 5–6 are close to neutral; 10 pulls the vector strongly toward
    that item; 1 pushes it away.

    Args:
        original_query_vector: L2-normalised query embedding from semantic_search.
        rated_items: Mapping of item title → user rating in [1, 10].
        candidate_pool: The full initial_pool returned by semantic_search.
                        Each item must contain "title" and "embedding" keys.
                        Items whose title appears in rated_items are used to
                        adjust the query vector; the rest are re-ranked.
        top_k: Number of results to return.

    Returns:
        Top-k items from the unrated portion of candidate_pool, re-ranked by
        cosine similarity against the refined query vector.  Each item contains
        all original fields except "embedding", plus "feedback_score" and
        "justification".
    """
    if not candidate_pool:
        return []

    rated_titles = {t.lower(): r for t, r in rated_items.items()}

    rated: list[dict] = []
    unrated: list[dict] = []
    for item in candidate_pool:
        if item.get("title", "").lower() in rated_titles:
            rated.append(item)
        else:
            unrated.append(item)

    if not rated or not unrated:
        return _format_results(
            sorted(unrated or candidate_pool, key=lambda x: x.get("score", 0.0), reverse=True),
            np.zeros(1),  # placeholder — feedback_score will be 0.0
            list(range(min(top_k, len(unrated or candidate_pool)))),
            use_original_score=True,
        )

    # Build weight vector and embedding matrix for rated items.
    # Linear normalisation: w = (r - 5.5) / 4.5  maps [1, 10] → [-1.0, +1.0].
    weights = np.array(
        [(rated_titles[item["title"].lower()] - 5.5) / 4.5 for item in rated],
        dtype=np.float32,
    )                                                           # (n_rated,)
    rated_embs = np.array(
        [item["embedding"] for item in rated], dtype=np.float32
    )                                                           # (n_rated, d)

    # Rocchio adjustment: weighted sum of rated embeddings added to query.
    # weights @ rated_embs contracts (n_rated,) × (n_rated, d) → (d,).
    adjustment = weights @ rated_embs
    contextual_query_vector = original_query_vector.astype(np.float32) + adjustment

    norm = np.linalg.norm(contextual_query_vector)
    if norm > 0:
        contextual_query_vector /= norm

    # Cosine similarity: normalise pool embeddings then dot with refined query.
    pool_embs = np.array(
        [item["embedding"] for item in unrated], dtype=np.float32
    )                                                           # (n_pool, d)
    pool_norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
    pool_normed = pool_embs / np.where(pool_norms == 0, 1.0, pool_norms)

    feedback_scores = pool_normed @ contextual_query_vector    # (n_pool,)

    top_indices = np.argsort(feedback_scores)[::-1][:top_k]
    return _format_results(unrated, feedback_scores, top_indices)


def _format_results(
    pool: list[dict],
    scores: np.ndarray,
    indices: list[int],
    use_original_score: bool = False,
) -> list[dict]:
    results = []
    for i in indices:
        item = pool[i]
        entry = {k: v for k, v in item.items() if k != "embedding"}
        entry["feedback_score"] = item.get("score", 0.0) if use_original_score else float(scores[i])
        entry["justification"] = _build_justification(item)
        results.append(entry)
    return results
