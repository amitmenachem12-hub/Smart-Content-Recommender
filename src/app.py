import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from relevance_feedback import apply_user_feedback
from search_engine import load_model, load_movies, semantic_search

_POOL_SIZE = 100
_TOP_K = 10
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


# ---------------------------------------------------------------------------
# Cached resource loaders — initialised once per Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_model():
    return load_model()


@st.cache_data
def _get_movies() -> list[dict]:
    return load_movies()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^\w]", "_", text)


def _display_label(item: dict) -> str:
    return f"{item['title']} ({item.get('media_type', '?')})"


def _genre_names(item: dict) -> list[str]:
    return [g["name"] if isinstance(g, dict) else g for g in item.get("genres", [])]


def _render_horizontal_card(item: dict, rank: int) -> None:
    """Wide horizontal card for Stage 3: poster on the left, details on the right."""
    score = item.get("feedback_score", item.get("score", 0.0))
    genres = _genre_names(item)

    with st.container(border=True):
        left, right = st.columns([1, 4])
        with left:
            path = item.get("poster_path")
            if path:
                full_url = f"{_TMDB_IMAGE_BASE}/{path.lstrip('/')}"
                try:
                    st.image(full_url, use_container_width=True)
                except Exception:
                    st.info("Image unavailable")
            else:
                st.info("Image unavailable")
        with right:
            st.markdown(f"### {rank}. {item['title']}")
            st.caption(f"`{item.get('media_type', '?')}` · Match score: **{score:.2f}**")
            if genres:
                st.caption(" · ".join(genres))
            st.write(item.get("overview", ""))


def _render_result_card(item: dict, rank: int) -> None:
    """Compact card used for overflow results (rank > 5) in Stage 3."""
    score = item.get("feedback_score", item.get("score", 0.0))
    genres = _genre_names(item)
    with st.container(border=True):
        cols = st.columns([8, 2])
        with cols[0]:
            st.markdown(f"**{rank}. {item['title']}**")
        with cols[1]:
            st.caption(f"`{item.get('media_type', '?')}` · {score:.3f}")
        if genres:
            st.caption(", ".join(genres))
        overview = item.get("overview", "")
        if len(overview) > 200:
            with st.expander("Overview"):
                st.write(overview)
        else:
            st.write(overview)
        if justification := item.get("justification"):
            st.info(justification, icon="💡")


# ---------------------------------------------------------------------------
# Session-state initialisation & reset
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "stage": 0,
        "query": "",
        "initial_pool": [],
        "top_k_results": [],
        "query_vector": None,
        "seen_labels": [],
        "final_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def _go_to(stage: int) -> None:
    st.session_state.stage = stage
    st.rerun()


# ---------------------------------------------------------------------------
# Stage renderers
# ---------------------------------------------------------------------------

def _stage_0() -> None:
    st.subheader("What are you in the mood for?")
    query = st.text_input(
        "Describe what you want to watch:",
        placeholder="e.g. A relaxing comedy about friends in New York",
        key="query_input",
    )
    if st.button("Search", type="primary", disabled=not query.strip()):
        with st.spinner("Running hybrid semantic search…"):
            result = semantic_search(
                query.strip(),
                top_k=_TOP_K,
                pool_size=_POOL_SIZE,
                model=_get_model(),
                movies_data=_get_movies(),
            )
        st.session_state.query = query.strip()
        st.session_state.initial_pool = result["initial_pool"]
        st.session_state.top_k_results = result["top_k_results"]
        st.session_state.query_vector = result["query_vector"]
        _go_to(1)


def _stage_1() -> None:
    st.subheader("Step 1 — Filter seen content")
    st.caption(f"Pool of **{len(st.session_state.initial_pool)}** results for: *{st.session_state.query}*")
    st.write(
        "Select anything you've already seen so we can learn your taste. "
        "Skip this step to get recommendations straight away."
    )

    options = [_display_label(item) for item in st.session_state.initial_pool]

    seen = st.multiselect(
        "Titles you've already seen:",
        options=options,
        key="seen_multiselect",
    )

    st.divider()
    left, _, right = st.columns([2, 4, 2])
    with left:
        if st.button("Start Over"):
            _reset()
    with right:
        next_label = "Rate Seen Items →" if seen else "Get Recommendations →"
        if st.button(next_label, type="primary"):
            st.session_state.seen_labels = seen
            if seen:
                _go_to(2)
            else:
                st.session_state.final_results = []
                _go_to(3)


def _stage_2() -> None:
    st.subheader("Step 2 — Rate what you've seen")
    st.caption(
        "Rate each title 1–10. "
        "Ratings pull recommendations toward (high) or away from (low) similar content."
    )

    label_to_item = {_display_label(item): item for item in st.session_state.initial_pool}

    ratings: dict[str, int] = {}
    for label in st.session_state.seen_labels:
        item = label_to_item.get(label)
        if item is None:
            continue
        rating = st.slider(
            label,
            min_value=1,
            max_value=10,
            value=5,
            key=f"rating_{_slug(item['title'])}",
        )
        ratings[item["title"]] = rating

    st.divider()
    left, mid, _, right = st.columns([2, 2, 2, 2])
    with left:
        if st.button("← Back"):
            _go_to(1)
    with mid:
        if st.button("Start Over"):
            _reset()
    with right:
        if st.button("Get Recommendations →", type="primary"):
            with st.spinner("Applying relevance feedback and re-ranking…"):
                final = apply_user_feedback(
                    original_query_vector=st.session_state.query_vector,
                    rated_items=ratings,
                    candidate_pool=st.session_state.initial_pool,
                    top_k=_TOP_K,
                )
            st.session_state.final_results = final
            _go_to(3)


def _stage_3() -> None:
    had_feedback = bool(st.session_state.seen_labels)

    if had_feedback:
        st.subheader("Your Personalised Recommendations")
        st.caption("Re-ranked via Rocchio relevance feedback on your ratings.")
        results = st.session_state.final_results
    else:
        st.subheader("Top Recommendations")
        st.caption(f"Semantic search results for: *{st.session_state.query}*")
        results = st.session_state.top_k_results

    if not results:
        st.warning("No recommendations found. Try a different query.")
    else:
        for rank, item in enumerate(results[:5], start=1):
            _render_horizontal_card(item, rank)
            if rank < min(5, len(results)):
                st.divider()

    st.divider()
    if st.button("Start Over", type="primary"):
        _reset()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Smart Content Recommender",
        page_icon="🎬",
        layout="centered",
    )
    st.title("🎬 Smart Content Recommender")

    _init_state()

    stage_renderers = {
        0: _stage_0,
        1: _stage_1,
        2: _stage_2,
        3: _stage_3,
    }
    stage_renderers[st.session_state.stage]()


if __name__ == "__main__":
    main()
