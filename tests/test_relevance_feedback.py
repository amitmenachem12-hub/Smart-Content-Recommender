"""Tests for apply_user_feedback (src/relevance_feedback.py).

Uses 3-D mock vectors — no sentence-transformer needed.

Rating scale: 1–5
  >= 4  →  positive  (pulls query toward the group's mean embedding, weight β=0.7)
  <= 2  →  negative  (pushes query away, weight γ=0.3)
  == 3  →  neutral   (ignored; triggers fallback to original score sort)

Rocchio formula:
  V_new = α·V_query + β·mean(V_positive) − γ·mean(V_negative),  then L2-normalised.
  Defaults: α=1.0, β=0.7, γ=0.3.

Fixture layout (base_pool)
--------------------------
  query  = [1, 0, 0]             (x-axis)
  "Pivot"  rated item;  emb = [0, 1, 0]   (y-axis)
  "Alpha"  close to Pivot:  [0.3, 0.7, 0]
  "Beta"   close to query:  [0.9, 0.0, 0.1]

  Positive (rating=5, β=0.7) → V_new = [1, 0.7, 0]/‖…‖
    Alpha ≈ 0.850,  Beta ≈ 0.814  →  Alpha first
  Negative (rating=1, γ=0.3) → V_new = [1, −0.3, 0]/‖…‖
    Alpha ≈ 0.113,  Beta ≈ 0.951  →  Beta first

Fixture layout (mixed_pool)
---------------------------
  query       = [1, 0, 0]
  "GoodPivot" rated +5; emb = [0, 1, 0]  (y-axis)
  "BadPivot"  rated  1; emb = [0, 0, 1]  (z-axis)
  V_new = [1, 0.7, −0.3]/‖…‖  →  MovieX(y)=0.557 > MovieY(z)=−0.239
"""

import numpy as np
import pytest

from src.relevance_feedback import apply_user_feedback


# ── helpers ──────────────────────────────────────────────────────────────────


def _item(title: str, emb: list[float], score: float = 0.5) -> dict:
    return {
        "title": title,
        "embedding": np.array(emb, dtype=np.float32),
        "score": score,
    }


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def base_pool():
    """query=[1,0,0]; Pivot rated; Alpha≈Pivot direction; Beta≈query direction."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pool = [
        _item("Pivot", [0.0, 1.0, 0.0], score=0.0),
        _item("Alpha", [0.3, 0.7, 0.0], score=0.3),
        _item("Beta",  [0.9, 0.0, 0.1], score=0.9),
    ]
    return query, pool


@pytest.fixture
def mixed_pool():
    """GoodPivot (y, rated +5) and BadPivot (z, rated 1) are both rated.
    MovieX≈liked, MovieY≈disliked, MovieZ≈original query direction."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pool = [
        _item("GoodPivot", [0.0, 1.0, 0.0], score=0.0),
        _item("BadPivot",  [0.0, 0.0, 1.0], score=0.1),
        _item("MovieX",    [0.0, 1.0, 0.0], score=0.2),
        _item("MovieY",    [0.0, 0.0, 1.0], score=0.3),
        _item("MovieZ",    [1.0, 0.0, 0.0], score=0.4),
    ]
    return query, pool


# ── positive feedback (rating >= 4) ──────────────────────────────────────────


class TestPositiveFeedback:

    def test_top_result_shifts_toward_liked_direction(self, base_pool):
        """rating=5 → V_new=[1,0.7,0]/norm; Alpha (closer to Pivot) ranks #1."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        assert results[0]["title"] == "Alpha"

    def test_liked_direction_score_exceeds_query_direction(self, base_pool):
        """Alpha's feedback_score must beat Beta's after a positive rating."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        by_title = {r["title"]: r["feedback_score"] for r in results}
        assert by_title["Alpha"] > by_title["Beta"]

    def test_minimum_positive_threshold_triggers_shift(self, base_pool):
        """rating=4 (minimum positive boundary) must also shift Alpha above Beta."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 4}, pool, top_k=2)
        assert results[0]["title"] == "Alpha"


# ── negative feedback (rating <= 2) ──────────────────────────────────────────


class TestNegativeFeedback:

    def test_top_result_shifts_away_from_disliked_direction(self, base_pool):
        """rating=1 → V_new=[1,−0.3,0]/norm; Beta (away from Pivot) ranks #1."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 1}, pool, top_k=2)
        assert results[0]["title"] == "Beta"

    def test_negative_depresses_score_of_similar_items(self, base_pool):
        """Alpha (similar to the disliked Pivot) must score lower with negative
        feedback than with positive feedback."""
        query, pool = base_pool
        pos = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        neg = apply_user_feedback(query, {"Pivot": 1}, pool, top_k=2)
        alpha_pos = next(r["feedback_score"] for r in pos if r["title"] == "Alpha")
        alpha_neg = next(r["feedback_score"] for r in neg if r["title"] == "Alpha")
        assert alpha_pos > alpha_neg

    def test_maximum_negative_threshold_triggers_shift(self, base_pool):
        """rating=2 (maximum negative boundary) must also push Beta to #1."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 2}, pool, top_k=2)
        assert results[0]["title"] == "Beta"


# ── mixed feedback ────────────────────────────────────────────────────────────


class TestMixedFeedback:

    def test_liked_direction_beats_disliked_direction(self, mixed_pool):
        """MovieX (≈liked y-axis) must outscore MovieY (≈disliked z-axis)
        after applying both positive and negative feedback."""
        query, pool = mixed_pool
        results = apply_user_feedback(
            query, {"GoodPivot": 5, "BadPivot": 1}, pool, top_k=3
        )
        by_title = {r["title"]: r["feedback_score"] for r in results}
        assert by_title["MovieX"] > by_title["MovieY"]

    def test_disliked_direction_item_ranks_last(self, mixed_pool):
        """MovieY (in the disliked z-direction) must be the bottom result."""
        query, pool = mixed_pool
        results = apply_user_feedback(
            query, {"GoodPivot": 5, "BadPivot": 1}, pool, top_k=3
        )
        assert results[-1]["title"] == "MovieY"


# ── edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_pool_returns_empty_list(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert apply_user_feedback(query, {"X": 5}, [], top_k=5) == []

    def test_empty_rated_dict_falls_back_to_original_scores(self, base_pool):
        """No rated_items → every item is unrated → fallback sorts by 'score' field."""
        query, pool = base_pool
        results = apply_user_feedback(query, {}, pool, top_k=3)
        assert [r["title"] for r in results] == ["Beta", "Alpha", "Pivot"]

    def test_all_neutral_ratings_falls_back_to_original_scores(self, base_pool):
        """rating=3 for every rated item → no positive/negative group → fallback."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 3}, pool, top_k=2)
        assert [r["title"] for r in results] == ["Beta", "Alpha"]

    def test_all_items_rated_falls_back_gracefully(self, base_pool):
        """All candidates rated → unrated pool is empty → fallback to original scores."""
        query, pool = base_pool
        rated = {item["title"]: 5 for item in pool}
        results = apply_user_feedback(query, rated, pool, top_k=3)
        assert [r["title"] for r in results] == ["Beta", "Alpha", "Pivot"]

    def test_only_negative_no_positive_does_not_crash(self, base_pool):
        """Only negative ratings → beta term is skipped; must return valid results."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 1}, pool, top_k=2)
        assert len(results) == 2
        assert all("feedback_score" in r for r in results)

    def test_only_positive_no_negative_does_not_crash(self, base_pool):
        """Only positive ratings → gamma term is skipped; must return valid results."""
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        assert len(results) == 2

    def test_output_omits_embedding_key(self, base_pool):
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        for item in results:
            assert "embedding" not in item

    def test_output_contains_required_keys(self, base_pool):
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=2)
        for item in results:
            assert "feedback_score" in item
            assert "justification" in item

    def test_top_k_limits_result_count(self, base_pool):
        query, pool = base_pool
        results = apply_user_feedback(query, {"Pivot": 5}, pool, top_k=1)
        assert len(results) == 1
