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


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def build_feedback_vector(
    original_query_vector: np.ndarray,
    rated_items: dict[str, int],
    candidate_pool: list[dict],
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.3,
) -> np.ndarray | None:
    """
    Compute the Rocchio-shifted, L2-normalised query vector.

        V_new = α · V_query + β · Σ(wᵢ · V_posᵢ) / |pos| − γ · mean(V_neg)

    Positive weight wᵢ = (rating − 3) / 2, so a 5-star pulls at full β and a
    4-star at half. Ratings <= 2 are negative, 3 is neutral (ignored).

    Returns None when no rated item carries a positive or negative signal.
    """
    rated_titles = {t.lower(): r for t, r in rated_items.items()}
    positive: list[tuple[dict, int]] = []
    negative: list[dict] = []
    for item in candidate_pool:
        rating = rated_titles.get(item.get("title", "").lower())
        if rating is None:
            continue
        if rating >= _POSITIVE_THRESHOLD:
            positive.append((item, rating))
        elif rating <= _NEGATIVE_THRESHOLD:
            negative.append(item)

    if not positive and not negative:
        return None

    vector = alpha * _unit(np.asarray(original_query_vector, dtype=np.float32))

    # Stored embeddings are not unit length (norm ≈ 3.5 with the current
    # model), so without normalising them β and γ would silently be ~3.5x
    # larger than configured and the original query would barely count.
    if positive:
        pos_embs = _unit(np.array([item["embedding"] for item, _ in positive], dtype=np.float32))
        weights = np.array([(r - 3) / 2 for _, r in positive], dtype=np.float32)
        vector = vector + beta * (weights[:, None] * pos_embs).sum(axis=0) / len(positive)

    if negative:
        neg_embs = _unit(np.array([item["embedding"] for item in negative], dtype=np.float32))
        vector = vector - gamma * neg_embs.mean(axis=0)

    return _unit(vector)


def apply_user_feedback(
    original_query_vector: np.ndarray,
    rated_items: dict[str, int],
    candidate_pool: list[dict],
    top_k: int = 5,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.3,
) -> list[dict]:
    """
    Re-rank the unrated part of candidate_pool against the Rocchio-shifted
    vector from build_feedback_vector.

    Falls back to the original "score" ordering when there are no rated items,
    no unrated items, or only neutral ratings.

    Returns:
        Top-k items, each with all original fields except "embedding", plus
        "feedback_score" and "justification".
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

    feedback_vector = build_feedback_vector(
        original_query_vector, rated_items, rated, alpha=alpha, beta=beta, gamma=gamma,
    )

    if feedback_vector is None:
        return _format_results(
            sorted(unrated, key=lambda x: x.get("score", 0.0), reverse=True),
            np.zeros(1),
            list(range(min(top_k, len(unrated)))),
            use_original_score=True,
        )

    pool_embs = _unit(np.array([item["embedding"] for item in unrated], dtype=np.float32))
    feedback_scores = pool_embs @ feedback_vector

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
