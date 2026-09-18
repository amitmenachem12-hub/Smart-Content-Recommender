import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

_BASE_URL = "https://api.themoviedb.org/3"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "movies_cache.json")

_COMMON_FIELDS = {"id", "overview", "genre_ids", "vote_average"}


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


def _fetch_genre_map() -> dict[int, str]:
    headers = _auth_headers()
    genre_map: dict[int, str] = {}
    for endpoint in ("/genre/movie/list", "/genre/tv/list"):
        response = requests.get(
            f"{_BASE_URL}{endpoint}",
            headers=headers,
            params={"language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        for g in response.json().get("genres", []):
            genre_map[g["id"]] = g["name"]
    return genre_map


def _normalize_item(raw: dict, media_type: str, genre_map: dict[int, str]) -> dict | None:
    overview = raw.get("overview") or ""
    if len(overview) < 15:
        return None
    title = raw.get("title") if media_type == "Movie" else raw.get("name")
    if not title:
        return None
    item = {k: raw[k] for k in _COMMON_FIELDS if k in raw}
    item["title"] = title
    item["genres"] = [genre_map[gid] for gid in item.pop("genre_ids", []) if gid in genre_map]
    item["media_type"] = media_type
    return item


def fetch_and_cache_popular_movies(limit: int = 2000) -> list[dict]:
    os.makedirs(_DATA_DIR, exist_ok=True)
    genre_map = _fetch_genre_map()
    headers = _auth_headers()
    seen: dict[tuple[str, int], dict] = {}

    sources = [
        ("/discover/movie", "Movie", 40),
        ("/discover/tv", "TV Show", 40),
    ]

    for endpoint, media_type, pages in sources:
        for page in range(1, pages + 1):
            response = requests.get(
                f"{_BASE_URL}{endpoint}",
                headers=headers,
                params={"page": page, "language": "en-US", "sort_by": "vote_count.desc"},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                break
            for raw in results:
                item_id = raw.get("id")
                key = (media_type, item_id)
                if item_id is not None and key not in seen:
                    item = _normalize_item(raw, media_type, genre_map)
                    if item is not None:
                        seen[key] = item

    content = list(seen.values())[:limit]
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)

    return content


if __name__ == "__main__":
    items = fetch_and_cache_popular_movies()
    print(f"Cached {len(items)} items to {os.path.abspath(_CACHE_PATH)}")
