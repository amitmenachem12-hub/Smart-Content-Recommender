import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

_BASE_URL = "https://api.themoviedb.org/3"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "movies_cache.json")


def _auth_headers() -> dict[str, str]:
    token = os.getenv("TMDB_ACCESS_TOKEN")
    if not token:
        raise EnvironmentError("TMDB_ACCESS_TOKEN is not set in the environment.")
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}


def search_movies(query: str, page: int = 1) -> list[dict]:
    response = requests.get(
        f"{_BASE_URL}/search/movie",
        headers=_auth_headers(),
        params={"query": query, "page": page, "language": "en-US"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("results", [])


_POPULAR_FIELDS = {"id", "title", "overview", "genre_ids", "vote_average"}


def fetch_and_cache_popular_movies(limit: int = 100) -> list[dict]:
    os.makedirs(_DATA_DIR, exist_ok=True)

    movies: list[dict] = []
    for page in range(1, 6):
        response = requests.get(
            f"{_BASE_URL}/movie/popular",
            headers=_auth_headers(),
            params={"page": page, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        movies.extend({k: m[k] for k in _POPULAR_FIELDS if k in m} for m in results)
        if len(movies) >= limit:
            break

    movies = movies[:limit]
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=4)

    return movies


if __name__ == "__main__":
    movies = fetch_and_cache_popular_movies()
    print(f"Cached {len(movies)} movies to {os.path.abspath(_CACHE_PATH)}")
