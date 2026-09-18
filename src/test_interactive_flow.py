import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_engine import semantic_search
from relevance_feedback import apply_user_feedback

_POOL_SAMPLE_SIZE = 100
_DIVIDER = "-" * 60


def _print_result(rank: int, item: dict, score_key: str = "score") -> None:
    score = item.get(score_key, 0.0)
    media = item.get("media_type", "Unknown")
    print(f"{rank}. {item['title']} [{media}] ({score:.4f})")
    overview = item.get("overview", "")
    print(f"   {overview[:120]}{'...' if len(overview) > 120 else ''}")
    if "justification" in item:
        print(f"   {item['justification']}")
    print()


def _prompt_rating(title: str) -> int | None:
    while True:
        raw = input(f"  How would you rate '{title}' (1-10)? ").strip()
        if not raw:
            print(f"  Skipping '{title}'.")
            return None
        try:
            rating = int(raw)
            if 1 <= rating <= 10:
                return rating
            print("  Please enter a number between 1 and 10.")
        except ValueError:
            print("  Please enter a whole number.")


def main() -> None:
    query = input("Enter your search query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return

    print("\nSearching...\n")
    output = semantic_search(query, top_k=5, pool_size=100)

    top_k_results = output["top_k_results"]
    initial_pool = output["initial_pool"]
    query_vector = output["query_vector"]

    print(_DIVIDER)
    print(f"Baseline Top 5 — query: '{query}'")
    print(_DIVIDER)
    if not top_k_results:
        print("No highly relevant matches found for your exact request.")
        return
    for rank, item in enumerate(top_k_results, start=1):
        _print_result(rank, item, score_key="score")

    if len(initial_pool) < 2:
        print("Candidate pool too small for feedback. Exiting.")
        return

    sample = initial_pool[:_POOL_SAMPLE_SIZE]
    print(_DIVIDER)
    print(f"Candidate pool ({len(sample)} items) — enter numbers of shows you have already seen:")
    print(_DIVIDER)
    for i, item in enumerate(sample, start=1):
        media = item.get("media_type", "Unknown")
        print(f"  {i:2}. {item['title']} [{media}]")

    print()
    raw_selection = input("Numbers of shows you've seen (e.g. 1, 4, 15) — or Enter to skip: ").strip()
    if not raw_selection:
        print("No feedback provided. Exiting.")
        return

    selected_indices: list[int] = []
    for part in raw_selection.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
            if 1 <= idx <= len(sample):
                selected_indices.append(idx - 1)
            else:
                print(f"  Ignoring {idx}: out of range.")
        except ValueError:
            print(f"  Ignoring '{part}': not a number.")

    if not selected_indices:
        print("No valid selections. Exiting.")
        return

    print()
    rated_items: dict[str, int] = {}
    for idx in selected_indices:
        title = sample[idx]["title"]
        rating = _prompt_rating(title)
        if rating is not None:
            rated_items[title] = rating

    if not rated_items:
        print("\nNo ratings collected. Exiting.")
        return

    print(f"\nRatings: {rated_items}\n")
    print("Re-ranking...\n")

    reranked = apply_user_feedback(query_vector, rated_items, initial_pool, top_k=5)

    print(_DIVIDER)
    print("Re-ranked Top 5 — after relevance feedback")
    print(_DIVIDER)
    if not reranked:
        print("No results after re-ranking.")
        return
    for rank, item in enumerate(reranked, start=1):
        _print_result(rank, item, score_key="feedback_score")


if __name__ == "__main__":
    main()
