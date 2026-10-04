"""Tests for search_by_vector (src/search_engine.py) against a fake collection."""

import json

import numpy as np

from src.search_engine import search_by_vector


def _meta(title: str, media_type: str = "TV Show", genres: list[str] | None = None,
          certification: str = "TV-14") -> dict:
    return {
        "title": title,
        "overview": "",
        "media_type": media_type,
        "genres_json": json.dumps(genres or ["Comedy"]),
        "vote_average": 7.0,
        "poster_path": "",
        "watch_providers_json": "[]",
        "certification": certification,
        "original_language": "en",
        "origin_country_json": "[]",
    }


class _FakeCollection:
    """Returns rows in the given order regardless of the query vector."""

    def __init__(self, rows: list[tuple[str, dict]]):
        self.rows = rows
        self.last_kwargs: dict = {}

    def count(self) -> int:
        return len(self.rows)

    def query(self, **kwargs) -> dict:
        self.last_kwargs = kwargs
        rows = self.rows
        where = kwargs.get("where")
        if where:
            rows = [r for r in rows if r[1]["media_type"] == where["media_type"]["$eq"]]
        rows = rows[: kwargs["n_results"]]
        return {
            "ids": [[r[0] for r in rows]],
            "distances": [[0.1 * i for i in range(len(rows))]],
            "metadatas": [[r[1] for r in rows]],
        }


VEC = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_excludes_phase1_ids_and_near_duplicate_titles():
    coll = _FakeCollection([
        ("TV Show_1", _meta("Friends")),
        ("TV Show_2", _meta("The Office")),       # duplicate entry under another ID
        ("TV Show_3", _meta("New Girl")),
        ("TV Show_4", _meta("How I Met Your Mother")),
    ])
    phase1 = [{"id": "TV Show_1", "title": "Friends"}, {"id": "TV Show_9", "title": "The Office"}]
    recs = search_by_vector(VEC, "funny show", 10, phase1, collection=coll)
    assert [r["title"] for r in recs] == ["New Girl", "How I Met Your Mother"]


def test_applies_original_query_hard_filters():
    coll = _FakeCollection([
        ("Movie_1", _meta("Some Film", media_type="Movie")),
        ("TV Show_2", _meta("Dark Drama", genres=["Drama"])),
        ("TV Show_3", _meta("Edgy Comedy", certification="TV-MA")),
        ("TV Show_4", _meta("New Girl")),
    ])
    recs = search_by_vector(VEC, "funny show", 10, [], collection=coll, safe_search=True)
    assert [r["title"] for r in recs] == ["New Girl"]
    assert coll.last_kwargs["where"] == {"media_type": {"$eq": "TV Show"}}


def test_respects_n_results_and_adds_score_and_justification():
    coll = _FakeCollection([(f"TV Show_{i}", _meta(f"Show{i}")) for i in range(10)])
    recs = search_by_vector(VEC, "something", 3, [], collection=coll)
    assert len(recs) == 3
    assert recs[0]["score"] == 1.0
    assert all("justification" in r and "embedding" not in r for r in recs)
