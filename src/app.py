import concurrent.futures
import html as html_module
import json
import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from relevance_feedback import build_feedback_vector
from search_engine import (
    get_hybrid_recommendations,
    load_collection,
    load_model,
    semantic_search,
)
from tmdb_client import fetch_item_details, fetch_tmdb_recommendations
from translations import UI_TEXT

_POOL_SIZE = 100
_TOP_K = 10
_PHASE1_K = 24
# Fetched deeper than _TOP_K so peers survive the frontend streaming-provider filter.
_PHASE2_DEPTH = 40
_ENRICHED_KEYS = ("title_en", "title_he", "overview_en", "overview_he", "watch_providers", "trailer_key")
_GRID_COLS = 3
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
_TMDB_LOGO_BASE = "https://image.tmdb.org/t/p/w45"


@st.cache_resource
def _get_model():
    return load_model()


@st.cache_resource
def _get_collection():
    return load_collection()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^\w]", "_", text)


def _genre_names(item: dict) -> list[str]:
    return [g["name"] if isinstance(g, dict) else g for g in item.get("genres", [])]


def _esc(value: object) -> str:
    return html_module.escape(str(value))


# ─── CSS ─────────────────────────────────────────────────────────────────────

_STYLES = """
<style>
/* ═══════════════════════════════════════════════════════════════
   Streaming-app design system
   ═══════════════════════════════════════════════════════════════ */

/* ── Hero ──────────────────────────────────────────────────────── */
.scr-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 62vh;
    text-align: center;
    padding: 60px 24px;
    background: radial-gradient(ellipse 80% 55% at 50% 0%,
                    rgba(124,58,237,.18) 0%, transparent 72%);
    border-radius: 20px;
}
.scr-hero-icon {
    font-size: 68px;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 28px rgba(124,58,237,.55));
}
.scr-hero-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(135deg, #E2D9F3 0%, #A78BFA 45%, #7C3AED 85%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 14px;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.scr-hero-subtitle {
    font-size: 18px;
    color: rgba(226,232,240,.52);
    margin: 0;
    max-width: 420px;
    line-height: 1.65;
}

/* ── Providers ─────────────────────────────────────────────────── */
.scr-providers {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 2px;
}
.scr-providers-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(167,139,250,.52);
    text-transform: uppercase;
    letter-spacing: .6px;
    margin-right: 2px;
}
.scr-provider-logo {
    width: 24px;
    height: 24px;
    border-radius: 4px;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,.08);
}
.scr-provider-text { font-size: 12px; color: rgba(226,232,240,.52); }

.scr-sidebar-query {
    background: rgba(124,58,237,.11);
    border: 1px solid rgba(124,58,237,.22);
    border-radius: 8px;
    padding: 9px 13px;
    font-size: 14px;
    font-style: italic;
    color: #E2E8F0;
    margin: 8px 0;
    word-break: break-word;
    line-height: 1.5;
}

/* ── Footer ─────────────────────────────────────────────────────── */
.scr-footer {
    margin-top: 16px;
    margin-bottom: 48px; /* clears Streamlit's fixed running-indicator area */
    text-align: center;
    font-size: 11px;
    color: rgba(226,232,240,.28);
    line-height: 1.5;
}

/* ── Mobile ─────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .scr-hero-title { font-size: 36px; }
}
</style>
"""


def _inject_styles() -> None:
    st.html(_STYLES)


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def _provider_logos_html(item: dict) -> str:
    providers = item.get("watch_providers", [])
    if not providers:
        return ""
    logos, names = [], []
    for p in providers[:6]:
        logo = p.get("logo_path", "")
        name = _esc(p.get("name", ""))
        if logo:
            url = f"{_TMDB_LOGO_BASE}/{logo.lstrip('/')}"
            logos.append(
                f'<img src="{url}" title="{name}" class="scr-provider-logo" alt="{name}">'
            )
        else:
            names.append(name)
    if not logos and not names:
        return ""
    inner = "".join(logos)
    if names:
        inner += f'<span class="scr-provider-text">{", ".join(names)}</span>'
    return (
        f'<div class="scr-providers">'
        f'<span class="scr-providers-label">Watch on</span>{inner}</div>'
    )


# ─── Trailer dialog ───────────────────────────────────────────────────────────

@st.dialog("Watch Trailer", width="large")
def _trailer_dialog() -> None:
    key = st.session_state.get("_trailer_key", "")
    if key:
        st.video(f"https://www.youtube.com/watch?v={key}")


# ─── Card rendering ───────────────────────────────────────────────────────────

def _render_trailer_button(item: dict, rank: int) -> None:
    trailer_key = item.get("trailer_key")
    if not trailer_key:
        return
    item_slug = _slug(str(item.get("id", rank)))
    if st.button(
        ":material/play_circle: Watch Trailer",
        key=f"trailer_{item_slug}",
        width="stretch",
    ):
        st.session_state["_trailer_key"] = trailer_key
        _trailer_dialog()


def _render_grid_card(item: dict, rank: int, rating_key: str) -> None:
    """Vertically stacked card sized for a narrow grid column (Phase 1)."""
    lang = st.session_state.get("lang", "en")
    title = item.get(f"title_{lang}") or item.get("title", "")
    overview = item.get(f"overview_{lang}") or item.get("overview", "")
    synopsis = (overview[:90] + "…") if len(overview) > 90 else overview
    media_type = item.get("media_type", "?")
    type_color = "blue" if media_type == "tv" else "violet"

    with st.container(border=True):
        path = item.get("poster_path", "")
        if path:
            st.image(f"{_TMDB_IMAGE_BASE}/{path.lstrip('/')}", width="stretch")
        else:
            st.markdown(":material/movie:")
        st.markdown(f"**#{rank} · {title}** &nbsp;:{type_color}[{media_type.upper()}]")
        if synopsis:
            st.caption(synopsis)
        st.feedback(options="stars", key=rating_key)
        _render_trailer_button(item, rank)


def _render_card(item: dict, rank: int) -> None:
    lang = st.session_state.get("lang", "en")
    title = item.get(f"title_{lang}") or item.get("title", "")
    overview = item.get(f"overview_{lang}") or item.get("overview", "")
    synopsis = (overview[:130] + "…") if len(overview) > 130 else overview
    media_type = item.get("media_type", "?")
    genres = _genre_names(item)
    type_color = "blue" if media_type == "tv" else "violet"

    with st.container(border=True):
        img_col, info_col = st.columns([1, 3])
        with img_col:
            path = item.get("poster_path", "")
            if path:
                st.image(f"{_TMDB_IMAGE_BASE}/{path.lstrip('/')}")
            else:
                st.markdown(":material/movie:")
        with info_col:
            st.markdown(
                f"**#{rank} · {title}** &nbsp;:{type_color}[{media_type.upper()}]"
            )
            if genres:
                st.caption(" · ".join(genres[:3]))
            if synopsis:
                st.caption(synopsis)
            providers_html = _provider_logos_html(item)
            if providers_html:
                st.html(providers_html)
            else:
                st.caption(":material/block: Not available for streaming in your region")
            if item.get("justification"):
                g_str = " · ".join(genres[:2]) if genres else ""
                st.caption(
                    f":material/auto_awesome: {g_str}"
                    if g_str
                    else ":material/auto_awesome: Strong semantic match"
                )
            _render_trailer_button(item, rank)


# ─── Session state ────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "stage": 0,
        "query": "",
        "safe_search": False,
        "initial_pool": [],
        "top_k_results": [],
        "query_vector": None,
        "final_recs": [],
        "lang": "en",
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


# ─── Enrichment ───────────────────────────────────────────────────────────────

def _enrich_with_translations(items: list[dict]) -> list[dict]:
    """Fetch translations + IL watch providers from TMDB for each item."""
    def _enrich_one(item: dict) -> dict:
        numeric_id = int(str(item["id"]).rsplit("_", 1)[-1])
        details = fetch_item_details(numeric_id, item["media_type"])
        return {
            **item,
            "title_en": details.get("title_en") or item.get("title", ""),
            "title_he": details.get("title_he") or item.get("title", ""),
            "overview_en": details.get("overview_en") or item.get("overview", ""),
            "overview_he": details.get("overview_he") or item.get("overview", ""),
            "watch_providers": details.get("watch_providers", item.get("watch_providers", [])),
            "trailer_key": details.get("trailer_key"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(_enrich_one, items))


def _merge_enrichment(items: list[dict], enriched: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in enriched}
    return [
        {**item, **{k: by_id[item["id"]][k] for k in _ENRICHED_KEYS if k in by_id.get(item["id"], {})}}
        for item in items
    ]


# ─── Debug helpers ────────────────────────────────────────────────────────────

def _items_from_ids(ids, collection) -> list[dict]:
    """Fetch minimal item dicts (id/title/media_type/genres) for a list of composite IDs."""
    if not ids:
        return []
    result = collection.get(ids=ids, include=["metadatas"])
    return [
        {
            "id": doc_id,
            "title": meta.get("title", ""),
            "media_type": meta.get("media_type", ""),
            "genres": json.loads(meta.get("genres_json", "[]")),
        }
        for doc_id, meta in zip(result["ids"], result["metadatas"])
    ]


def _chroma_items_for_vector(query_vec, collection, exclude_ids: set, n: int = 60) -> list[dict]:
    """Top-n ChromaDB hits (phase-1 items excluded) as full item dicts."""
    fetch_n = min(n, collection.count())
    if fetch_n == 0:
        return []
    result = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=fetch_n,
        include=["metadatas"],
    )
    items = []
    for doc_id, meta in zip(result["ids"][0], result["metadatas"][0]):
        if doc_id in exclude_ids:
            continue
        items.append({
            "id": doc_id,
            "title": meta.get("title", ""),
            "overview": meta.get("overview", ""),
            "media_type": meta.get("media_type", ""),
            "genres": json.loads(meta.get("genres_json", "[]")),
            "vote_average": meta.get("vote_average", 0.0),
            "poster_path": meta.get("poster_path", ""),
            "watch_providers": json.loads(meta.get("watch_providers_json", "[]")),
            "certification": meta.get("certification", "Unrated"),
            "original_language": meta.get("original_language", ""),
            "origin_country": json.loads(meta.get("origin_country_json", "[]")),
        })
    return items


# ─── Rating helpers ───────────────────────────────────────────────────────────

def _fetch_collab_rec_ids(items: list[dict], ratings: dict[str, int]) -> list[str]:
    """Return deduplicated ChromaDB composite IDs from TMDB's /recommendations endpoint
    for every item the user rated 4 or 5 stars.

    /recommendations uses TMDB's collaborative-filtering graph, which captures actual
    viewing patterns (e.g. Friends watchers also watch HIMYM) rather than keyword/genre
    similarity.  20 candidates per anchor give the blocklist veto enough material to
    work with while keeping API latency low.
    """
    highly_rated = [item for item in items if ratings.get(item["title"], 0) >= 4]
    if not highly_rated:
        return []

    def _recs_for(item: dict) -> list[str]:
        numeric_id = int(str(item["id"]).rsplit("_", 1)[-1])
        prefix = "Movie" if item["media_type"] == "Movie" else "TV Show"
        rec_ids = fetch_tmdb_recommendations(numeric_id, item["media_type"], max_results=20)
        return [f"{prefix}_{rid}" for rid in rec_ids]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        batches = list(ex.map(_recs_for, highly_rated))

    seen: set[str] = set()
    result: list[str] = []
    for batch in batches:
        for cid in batch:
            if cid not in seen:
                result.append(cid)
                seen.add(cid)
    return result


def _anchor_genres(items: list[dict], ratings: dict[str, int]) -> set[str]:
    """Primary genre name of each item rated 4 or 5 stars.
    Used as the genre gate in _validate_tmdb_recs."""
    genres: set[str] = set()
    for item in items:
        if ratings.get(item["title"], 0) < 4:
            continue
        item_genres = [g["name"] if isinstance(g, dict) else g for g in item.get("genres", [])]
        if item_genres:
            genres.add(item_genres[0])
    return genres


def _anchor_all_genres(items: list[dict], ratings: dict[str, int]) -> set[str]:
    """ALL genre names from items rated 4 or 5 stars.
    Used for the free-pass check: a TMDB rec whose entire genre list is a
    subset of this set skips the vector similarity threshold."""
    genres: set[str] = set()
    for item in items:
        if ratings.get(item["title"], 0) < 4:
            continue
        genres.update(
            g["name"] if isinstance(g, dict) else g
            for g in item.get("genres", [])
        )
    return genres


def _collect_ratings() -> dict[str, int]:
    """Read all star ratings from session state; returns title → 1-5 score."""
    ratings: dict[str, int] = {}
    for item in st.session_state.get("initial_pool", []):
        val = st.session_state.get(f"rating_{_slug(str(item['id']))}")
        if val is not None:
            ratings[item["title"]] = val + 1  # st.feedback 0-indexed (0–4) → 1–5
    return ratings


def _header_row(t: dict) -> None:
    """Query pill + start-over button, shared by stage 1 and 2."""
    hdr_col, reset_col = st.columns([5, 1])
    with hdr_col:
        st.html(
            f'<div class="scr-sidebar-query">"{_esc(st.session_state.query)}"</div>'
        )
    with reset_col:
        st.space("small")
        if st.button(t["start_over"], key="header_reset"):
            _reset()


def _provider_filter(display_recs: list[dict], t: dict) -> list[dict]:
    """Render provider multiselect and return the filtered list."""
    all_providers = sorted({
        p["name"]
        for item in display_recs
        for p in item.get("watch_providers", [])
        if p.get("name")
    })
    if not all_providers:
        return display_recs

    if "provider_filter" in st.session_state:
        st.session_state.provider_filter = [
            p for p in st.session_state.provider_filter if p in all_providers
        ]
    st.multiselect(
        t["provider_filter_label"],
        options=all_providers,
        key="provider_filter",
        placeholder=t["provider_filter_placeholder"],
    )

    selected: list[str] = st.session_state.get("provider_filter", [])
    if selected:
        display_recs = [
            item for item in display_recs
            if any(p["name"] in selected for p in item.get("watch_providers", []))
        ]
    return display_recs


# ─── Stage renderers ─────────────────────────────────────────────────────────

def _main_stage_0() -> None:
    t = UI_TEXT[st.session_state.lang]
    st.html(f"""
<div class="scr-hero">
  <div class="scr-hero-icon">🎬</div>
  <h1 class="scr-hero-title">{_esc(t["hero_title"])}</h1>
  <p class="scr-hero-subtitle">{_esc(t["hero_subtitle"])}</p>
</div>
""")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("search_form"):
            query = st.text_input(
                t["search_label"],
                placeholder=t["search_placeholder"],
                key="query_input",
            )
            safe_search = st.toggle(
                "Family-friendly only",
                value=False,
                key="safe_search_toggle",
            )
            st.space("small")
            submitted = st.form_submit_button(
                ":material/search: Search",
                type="primary",
            )
        if submitted and query.strip():
            with st.spinner("Searching…"):
                result = semantic_search(
                    query.strip(),
                    top_k=_PHASE1_K,
                    pool_size=_POOL_SIZE,
                    model=_get_model(),
                    collection=_get_collection(),
                    safe_search=safe_search,
                )
                result["initial_pool"] = _enrich_with_translations(result["initial_pool"])
                result["top_k_results"] = _merge_enrichment(
                    result["top_k_results"], result["initial_pool"]
                )
            st.session_state.query = query.strip()
            st.session_state.safe_search = safe_search
            st.session_state.initial_pool = result["initial_pool"]
            st.session_state.top_k_results = result["top_k_results"]
            st.session_state.query_vector = result["query_vector"]
            st.session_state.final_recs = []
            st.session_state.provider_filter = []
            _go_to(1)


def _main_stage_1() -> None:
    """Phase 1: show initial candidates for rating. Rocchio runs on button click."""
    t = UI_TEXT[st.session_state.lang]
    _header_row(t)

    st.markdown(f"#### {t['rate_section_title']}")
    st.caption(t["rate_section_caption"])
    st.space("small")

    items = st.session_state.top_k_results
    # Build row by row (a fresh st.columns per row) so cards in the same row
    # stay top-aligned even when synopsis lengths differ.
    for row_start in range(0, len(items), _GRID_COLS):
        cols = st.columns(_GRID_COLS)
        for offset, item in enumerate(items[row_start:row_start + _GRID_COLS]):
            with cols[offset]:
                _render_grid_card(
                    item,
                    rank=row_start + offset + 1,
                    rating_key=f"rating_{_slug(str(item['id']))}",
                )

    st.divider()

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button(t["get_recs_button"], type="primary", key="get_recs"):
            with st.spinner("Personalizing…"):
                pool = st.session_state.initial_pool
                ratings = _collect_ratings()
                feedback_vector = build_feedback_vector(
                    st.session_state.query_vector, ratings, pool,
                )
                query_vec = feedback_vector if feedback_vector is not None else st.session_state.query_vector
                anchor_items = [it for it in items if ratings.get(it["title"], 0) >= 4]
                phase_1_ids = {it["id"] for it in items}
                coll = _get_collection()
                # Oversample ChromaDB: genre filter + RRF will trim the list down.
                chroma_items_raw = _chroma_items_for_vector(
                    query_vec, coll, phase_1_ids, n=_PHASE2_DEPTH * 3,
                )
                # get_hybrid_recommendations fetches /discover internally and fuses
                # via RRF; the sidebar debug expander fires as a side effect.
                recs = get_hybrid_recommendations(
                    chroma_items_raw, anchor_items, phase_1_ids, collection=coll,
                )
                pool_ids = {p["id"] for p in pool}
                # Pool items were already enriched in stage 0; only fetch the new ones.
                newly_enriched = _enrich_with_translations(
                    [r for r in recs if r["id"] not in pool_ids]
                )
                st.session_state.final_recs = _merge_enrichment(recs, pool + newly_enriched)
                st.session_state.debug_hybrid = {
                    "chroma_raw": [it["title"] for it in chroma_items_raw],
                }
            st.session_state.provider_filter = []
            _go_to(2)


def _main_stage_2() -> None:
    """Phase 2: show final personalized recommendations, all phase-1 IDs excluded."""
    t = UI_TEXT[st.session_state.lang]
    _header_row(t)

    st.markdown(f"#### {t['final_recs_title']}")
    ratings = _collect_ratings()
    n = len(ratings)
    if n:
        st.caption(t["final_recs_caption_rated"].format(n=n, s="" if n == 1 else "s"))
    else:
        st.caption(t["final_recs_caption_no_ratings"])
    st.space("small")

    display_recs = list(st.session_state.get("final_recs", []))

    if not display_recs:
        st.info("No recommendations found — try a different search.")
        return

    display_recs = _provider_filter(display_recs, t)

    if not display_recs:
        st.info(t["no_provider_matches"])
        return

    debug = st.session_state.get("debug_hybrid")
    if debug and "chroma_raw" in debug:
        with st.expander("Debug: Hybrid Engine Check"):
            st.write("**Raw ChromaDB Results** (before filter):", debug["chroma_raw"])
            st.caption("TMDB /discover results are shown in the sidebar expander during personalisation.")

    for rank, item in enumerate(display_recs[:_TOP_K], start=1):
        _render_card(item, rank=rank)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Smart Content Recommender",
        page_icon=":material/movie:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_styles()
    _init_state()

    _, lang_col = st.columns([4, 1])
    with lang_col:
        st.segmented_control(
            UI_TEXT[st.session_state.lang]["lang_toggle"],
            options=["en", "he"],
            default=st.session_state.lang,
            format_func=lambda x: x.upper(),
            key="lang",
        )

    if st.session_state.lang == "he":
        st.markdown("""
<style>
/* ── Hebrew RTL ─────────────────────────────────────────────────── */
.block-container, p, h1, h2, h3, div {
    direction: rtl;
    text-align: right;
}

/* Preserve st.columns order — RTL on a flex-row reverses item order */
[data-testid="stHorizontalBlock"] {
    direction: ltr;
}

/* Keep exception tracebacks and code blocks readable (LTR) */
.stException, .stException *,
code, pre {
    direction: ltr !important;
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

    main_fn = {
        0: _main_stage_0,
        1: _main_stage_1,
        2: _main_stage_2,
    }
    main_fn[st.session_state.stage]()

    st.html(
        f'<div class="scr-footer">{_esc(UI_TEXT[st.session_state.lang]["footer"])}</div>'
    )


if __name__ == "__main__":
    main()
