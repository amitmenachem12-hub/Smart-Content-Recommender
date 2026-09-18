import json
import os

from sentence_transformers import SentenceTransformer

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "movies_cache.json")
_OUTPUT_PATH = os.path.join(_DATA_DIR, "movies_with_embeddings.json")
_MODEL_NAME = "all-MiniLM-L6-v2"


def generate_movie_embeddings() -> None:
    with open(_CACHE_PATH, encoding="utf-8") as f:
        movies: list[dict] = json.load(f)

    model = SentenceTransformer(_MODEL_NAME)

    texts = [
        f"Type: {m.get('media_type') or ''}. Genres: {', '.join(m.get('genres') or [])}. Title: {m.get('title') or ''}. Overview: {m.get('overview') or ''}"
        for m in movies
    ]
    embeddings = model.encode(texts, show_progress_bar=True)

    movies_with_embeddings = [
        {**movie, "embedding": embedding.tolist()}
        for movie, embedding in zip(movies, embeddings)
    ]

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(movies_with_embeddings, f, indent=4)


if __name__ == "__main__":
    generate_movie_embeddings()
    print(f"Embeddings saved to {os.path.abspath(_OUTPUT_PATH)}")
