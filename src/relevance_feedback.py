import numpy as np

# Rating thresholds for the Rocchio classification step.
_POSITIVE_THRESHOLD: int = 4  # rating >= 4 → positive
_NEGATIVE_THRESHOLD: int = 2  # rating <= 2 → negative
# rating == 3 → neutral (ignored)


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
    alpha: float = 1.0,
    beta: float = 0.7,
    gamma: float = 0.3,
) -> list[dict]:
    """
    Refine results using Rocchio-style negative vector shifting.

    Rated items are classified by threshold:
      rating >= 4  →  positive: query pulled toward the group's mean embedding.
      rating <= 2  →  negative: query pushed away from the group's mean embedding.
      rating == 3  →  neutral:  ignored.

    The refined query vector is:
        V_new = α · V_query + β · mean(V_positive) − γ · mean(V_negative)
    then L2-normalised so it remains compatible with cosine-distance search.

    γ < β by default (0.3 vs 0.7) so negative signal is intentionally weaker
    than positive signal to avoid erratic semantic jumps.

    Args:
        original_query_vector: L2-normalised query embedding.
        rated_items: Mapping of item title → integer rating in [1, 5].
                     Title lookup is case-insensitive.
        candidate_pool: Items returned by semantic_search; each must contain
                        "title" and "embedding" keys.
        top_k: Number of results to return.
        alpha: Weight for the original query vector (default 1.0).
        beta:  Weight for the mean positive embedding (default 0.7).
        gamma: Weight for the mean negative embedding (default 0.3).

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
            np.zeros(1),
            list(range(min(top_k, len(unrated or candidate_pool)))),
            use_original_score=True,
        )

    positive = [
        item for item in rated
        if rated_titles[item["title"].lower()] >= _POSITIVE_THRESHOLD
    ]
    negative = [
        item for item in rated
        if rated_titles[item["title"].lower()] <= _NEGATIVE_THRESHOLD
    ]

    # All rated items were neutral — no actionable signal.
    if not positive and not negative:
        return _format_results(
            sorted(unrated, key=lambda x: x.get("score", 0.0), reverse=True),
            np.zeros(1),
            list(range(min(top_k, len(unrated)))),
            use_original_score=True,
        )

    contextual_query_vector: np.ndarray = alpha * original_query_vector.astype(np.float32)

    if positive:
        pos_embs = np.array([item["embedding"] for item in positive], dtype=np.float32)
        contextual_query_vector = contextual_query_vector + beta * pos_embs.mean(axis=0)

    if negative:
        neg_embs = np.array([item["embedding"] for item in negative], dtype=np.float32)
        contextual_query_vector = contextual_query_vector - gamma * neg_embs.mean(axis=0)

    norm = np.linalg.norm(contextual_query_vector)
    if norm > 0:
        contextual_query_vector /= norm

    pool_embs = np.array([item["embedding"] for item in unrated], dtype=np.float32)
    pool_norms = np.linalg.norm(pool_embs, axis=1, keepdims=True)
    pool_normed = pool_embs / np.where(pool_norms == 0, 1.0, pool_norms)

    feedback_scores = pool_normed @ contextual_query_vector

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
