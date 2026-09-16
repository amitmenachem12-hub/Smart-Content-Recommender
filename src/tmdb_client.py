import os
from dotenv import load_dotenv
import requests

load_dotenv()

_BASE_URL = "https://api.themoviedb.org/3"


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


if __name__ == "__main__":
    results = search_movies("Inception")
    if not results:
        print("No results found.")
    else:
        movie = results[0]
        year = (movie.get("release_date") or "")[:4] or "N/A"
        print(f"Title:    {movie['title']}")
        print(f"Year:     {year}")
        print(f"Overview: {movie.get('overview', 'N/A')}")
