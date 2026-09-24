import html as html_module
import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from relevance_feedback import apply_user_feedback
from search_engine import load_collection, load_model, semantic_search

_POOL_SIZE = 100
_TOP_K = 10
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
_TMDB_LOGO_BASE = "https://image.tmdb.org/t/p/w45"
_TMDB_THUMB_BASE = "https://image.tmdb.org/t/p/w92"


@st.cache_resource
def _get_model():
    return load_model()


@st.cache_resource
def _get_collection():
    return load_collection()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^\w]", "_", text)


def _display_label(item: dict) -> str:
    return f"{item['title']} ({item.get('media_type', '?')})"


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

/* ── Step progress strip ───────────────────────────────────────── */
.scr-steps {
    display: flex;
    margin-bottom: 24px;
    background: rgba(14,14,40,.65);
    border: 1px solid rgba(124,58,237,.14);
    border-radius: 10px;
    overflow: hidden;
}
.scr-step {
    flex: 1;
    text-align: center;
    padding: 10px 6px;
    font-size: 11px;
    font-weight: 600;
    color: rgba(226,232,240,.28);
    letter-spacing: .5px;
    text-transform: uppercase;
    border-right: 1px solid rgba(124,58,237,.1);
}
.scr-step:last-child { border-right: none; }
.scr-step.done  { color: rgba(196,181,253,.5); }
.scr-step.active {
    background: rgba(124,58,237,.18);
    color: #C4B5FD;
}

/* ── Results header ────────────────────────────────────────────── */
.scr-results-header { margin-bottom: 22px; }
.scr-results-title {
    font-size: 26px;
    font-weight: 700;
    color: #E2E8F0;
    margin: 0 0 4px;
}
.scr-results-meta {
    font-size: 14px;
    color: rgba(167,139,250,.72);
    margin: 0;
}

/* ── Card ──────────────────────────────────────────────────────── */
.scr-card {
    display: flex;
    background: rgba(12,12,38,.84);
    border-radius: 16px;
    border: 1px solid rgba(124,58,237,.18);
    overflow: hidden;
    margin-bottom: 20px;
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    box-shadow: 0 6px 28px rgba(0,0,0,.46);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    cursor: default;
}
.scr-card:hover {
    transform: scale(1.02);
    box-shadow: 0 18px 52px rgba(0,0,0,.65),
                0 4px 18px rgba(124,58,237,.30);
    border-color: rgba(124,58,237,.50);
}

/* ── Compact card override ─────────────────────────────────────── */
.scr-card.scr-compact {
    border-radius: 12px;
    margin-bottom: 14px;
    box-shadow: 0 3px 14px rgba(0,0,0,.38);
    background: rgba(12,12,38,.72);
    border-color: rgba(124,58,237,.13);
}
.scr-card.scr-compact:hover {
    box-shadow: 0 10px 28px rgba(0,0,0,.55),
                0 2px 10px rgba(124,58,237,.22);
    border-color: rgba(124,58,237,.40);
}

/* ── Poster ────────────────────────────────────────────────────── */
.scr-poster {
    width: 130px;
    min-width: 130px;
    height: 195px;
    object-fit: cover;
    flex-shrink: 0;
    display: block;
}
.scr-poster-ph {
    width: 130px;
    min-width: 130px;
    height: 195px;
    background: rgba(124,58,237,.09);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 36px;
    opacity: .45;
}
.scr-card.scr-compact .scr-poster,
.scr-card.scr-compact .scr-poster-ph {
    width: 82px;
    min-width: 82px;
    height: 123px;
}
.scr-card.scr-compact .scr-poster-ph { font-size: 26px; }

/* ── Card body ─────────────────────────────────────────────────── */
.scr-body {
    padding: 20px 24px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
}
.scr-card.scr-compact .scr-body { padding: 14px 18px; gap: 5px; }

/* ── Header row ────────────────────────────────────────────────── */
.scr-header {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    flex-wrap: wrap;
}
.scr-rank {
    font-size: 11px;
    font-weight: 700;
    color: #C4B5FD;
    background: rgba(124,58,237,.18);
    border-radius: 5px;
    padding: 3px 8px;
    flex-shrink: 0;
    margin-top: 3px;
    letter-spacing: .3px;
}
.scr-title {
    font-size: 20px;
    font-weight: 700;
    color: #F1F0FF;
    margin: 0;
    flex: 1;
    line-height: 1.3;
    min-width: 0;
    word-break: normal !important;
    overflow-wrap: anywhere !important;
}
.scr-card.scr-compact .scr-title { font-size: 15px; }
.scr-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
    flex-wrap: wrap;
}

/* ── Badge ─────────────────────────────────────────────────────── */
.scr-badge {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
    border-radius: 4px;
    padding: 2px 8px;
    background: rgba(124,58,237,.22);
    color: #C4B5FD;
    border: 1px solid rgba(124,58,237,.35);
}
.scr-badge-tv {
    background: rgba(59,130,246,.18);
    color: #93C5FD;
    border-color: rgba(59,130,246,.32);
}
.scr-score {
    font-size: 12px;
    font-weight: 600;
    color: rgba(226,232,240,.4);
    font-variant-numeric: tabular-nums;
}

/* ── Genres, overview ──────────────────────────────────────────── */
.scr-genres {
    font-size: 13px;
    color: rgba(196,181,253,.68);
    letter-spacing: .1px;
}
.scr-overview {
    font-size: 14px;
    color: rgba(226,232,240,.68);
    line-height: 1.65;
    margin: 0;
    flex: 1;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.scr-card.scr-compact .scr-overview {
    -webkit-line-clamp: 2;
    font-size: 13px;
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
    width: 26px;
    height: 26px;
    border-radius: 5px;
    object-fit: cover;
    border: 1px solid rgba(255,255,255,.08);
}
.scr-provider-text { font-size: 12px; color: rgba(226,232,240,.52); }

/* ── Justification callout ─────────────────────────────────────── */
.scr-just {
    font-size: 13px;
    color: rgba(196,181,253,.82);
    background: rgba(124,58,237,.10);
    border-left: 3px solid rgba(124,58,237,.52);
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
}

/* ── Pool thumbnail grid ───────────────────────────────────────── */
.scr-pool-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 4px 0 16px;
}
.scr-thumb {
    width: 70px;
    height: 105px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid rgba(124,58,237,.14);
    transition: transform .15s ease, border-color .15s ease;
    flex-shrink: 0;
    vertical-align: top;
}
.scr-thumb:hover {
    transform: scale(1.08);
    border-color: rgba(124,58,237,.5);
}
.scr-thumb-ph {
    width: 70px;
    height: 105px;
    border-radius: 8px;
    background: rgba(124,58,237,.08);
    border: 1px solid rgba(124,58,237,.12);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    opacity: .38;
    flex-shrink: 0;
    vertical-align: top;
}

/* ── Sidebar branding ──────────────────────────────────────────── */
.scr-brand {
    font-size: 20px;
    font-weight: 800;
    background: linear-gradient(90deg, #A78BFA 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.3px;
    display: block;
}
.scr-tagline {
    font-size: 12px;
    color: rgba(167,139,250,.52);
    display: block;
    margin-top: 2px;
}
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

/* ── Mobile typography ─────────────────────────────────────────── */
@media (max-width: 768px) {
    .scr-title {
        font-size: 1.1rem !important;
        flex-basis: 100%;
    }
}
</style>
"""


def _inject_styles() -> None:
    st.html(_STYLES)


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def _steps_html(active: int) -> str:
    labels = ["Search", "Filter", "Rate", "Results"]
    parts = []
    for i, label in enumerate(labels, 1):
        if i < active:
            cls = "scr-step done"
        elif i == active:
            cls = "scr-step active"
        else:
            cls = "scr-step"
        parts.append(f'<div class="{cls}">{label}</div>')
    return f'<div class="scr-steps">{"".join(parts)}</div>'


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


def _poster_img_html(item: dict) -> str:
    path = item.get("poster_path", "")
    title = _esc(item.get("title", ""))
    if path:
        url = f"{_TMDB_IMAGE_BASE}/{path.lstrip('/')}"
        return f'<img src="{url}" class="scr-poster" alt="{title} poster">'
    return '<div class="scr-poster-ph">🎬</div>'


def _card_html(item: dict, rank: int, compact: bool = False) -> str:
    score = item.get("feedback_score", item.get("score", 0.0))
    genres = _genre_names(item)
    title = _esc(item.get("title", ""))
    media_type = item.get("media_type", "?")
    overview = _esc(item.get("overview", ""))

    genre_html = (
        f'<div class="scr-genres">{" · ".join(_esc(g) for g in genres)}</div>'
        if genres else ""
    )
    just_html = ""
    if item.get("justification"):
        genre_str = ", ".join(genres) if genres else ""
        clean_just = (
            f"Recommended because it strongly matches your intent and features genres: {genre_str}"
            if genre_str
            else "Recommended because it strongly matches your intent"
        )
        just_html = f'<div class="scr-just">💡 {_esc(clean_just)}</div>'

    badge_cls = f"scr-badge scr-badge-{_esc(media_type)}"
    card_cls = "scr-card scr-compact" if compact else "scr-card"

    return f"""
<div class="{card_cls}">
  {_poster_img_html(item)}
  <div class="scr-body">
    <div class="scr-header">
      <span class="scr-rank">#{rank}</span>
      <span class="scr-title">{title}</span>
      <div class="scr-meta">
        <span class="{badge_cls}">{_esc(media_type)}</span>
        <span class="scr-score">{score:.2f}</span>
      </div>
    </div>
    {genre_html}
    <p class="scr-overview">{overview}</p>
    {_provider_logos_html(item)}
    {just_html}
  </div>
</div>"""


def _render_horizontal_card(item: dict, rank: int) -> None:
    st.html(_card_html(item, rank, compact=False))


def _render_result_card(item: dict, rank: int) -> None:
    st.html(_card_html(item, rank, compact=True))


# ─── Session state ────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "stage": 0,
        "query": "",
        "safe_search": False,
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


# ─── Main area ────────────────────────────────────────────────────────────────

def _pool_thumbs_html(items: list[dict], limit: int = 30) -> str:
    thumbs = []
    for item in items[:limit]:
        path = item.get("poster_path", "")
        title = _esc(item.get("title", ""))
        if path:
            url = f"{_TMDB_THUMB_BASE}/{path.lstrip('/')}"
            thumbs.append(
                f'<img src="{url}" title="{title}" class="scr-thumb" alt="{title}">'
            )
        else:
            thumbs.append('<span class="scr-thumb-ph">🎬</span>')
    return f'<div class="scr-pool-grid">{"".join(thumbs)}</div>'


def _main_stage_0() -> None:
    st.html("""
<div class="scr-hero">
  <div class="scr-hero-icon">🎬</div>
  <h1 class="scr-hero-title">Find your next binge.</h1>
  <p class="scr-hero-subtitle">
    Describe what you're in the mood for and our AI matches you to
    films and shows that fit — no genre checkboxes required.
  </p>
</div>
""")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        query = st.text_input(
            "What are you in the mood for?",
            placeholder="e.g. relaxing comedy about friends in New York",
            key="query_input",
        )
        safe_search = st.toggle(
            "Family-friendly only",
            value=False,
            key="safe_search_toggle",
        )
        st.space("small")
        if st.button(
            ":material/search: Search",
            type="primary",
            disabled=not query.strip(),
            key="search_btn",
        ):
            with st.spinner("Searching…"):
                result = semantic_search(
                    query.strip(),
                    top_k=_TOP_K,
                    pool_size=_POOL_SIZE,
                    model=_get_model(),
                    collection=_get_collection(),
                    safe_search=safe_search,
                )
            st.session_state.query = query.strip()
            st.session_state.safe_search = safe_search
            st.session_state.initial_pool = result["initial_pool"]
            st.session_state.top_k_results = result["top_k_results"]
            st.session_state.query_vector = result["query_vector"]
            _go_to(1)


def _main_stage_1() -> None:
    pool = st.session_state.initial_pool
    st.html(_steps_html(active=2))
    st.html(f"""
<div class="scr-results-header">
  <p class="scr-results-title">Found {len(pool)} matches</p>
  <p class="scr-results-meta">For &ldquo;{_esc(st.session_state.query)}&rdquo; — mark any titles you've already seen below.</p>
</div>
""")
    st.html(_pool_thumbs_html(pool))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.html(
            f'<div class="scr-sidebar-query">"{_esc(st.session_state.query)}"</div>'
        )
        st.caption(
            "Select titles you've already seen — we'll use your ratings to personalise results. "
            "Skip to get recommendations straight away."
        )
        options = [_display_label(item) for item in st.session_state.initial_pool]
        seen = st.multiselect(
            "Already seen:",
            options=options,
            key="seen_multiselect",
            label_visibility="visible",
        )
        st.space("small")
        next_label = ":material/star: Rate seen items" if seen else ":material/recommend: Get recommendations"
        if st.button(next_label, type="primary", key="stage1_next"):
            st.session_state.seen_labels = seen
            if seen:
                _go_to(2)
            else:
                st.session_state.final_results = []
                _go_to(3)
        if st.button(":material/restart_alt: Start over", key="stage1_reset"):
            _reset()


def _main_stage_2() -> None:
    seen_labels = st.session_state.seen_labels
    pool = st.session_state.initial_pool
    label_to_item = {_display_label(i): i for i in pool}

    st.html(_steps_html(active=3))
    st.html(f"""
<div class="scr-results-header">
  <p class="scr-results-title">Rating {len(seen_labels)} title{"s" if len(seen_labels) != 1 else ""}</p>
  <p class="scr-results-meta">Adjust the sliders below, then get your recommendations.</p>
</div>
""")
    rated_items = [label_to_item[l] for l in seen_labels if l in label_to_item]
    if rated_items:
        st.html(_pool_thumbs_html(rated_items, limit=len(rated_items)))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.html(
            f'<div class="scr-sidebar-query">"{_esc(st.session_state.query)}"</div>'
        )
        st.caption("Rate 1–10. High scores pull results toward similar content; low scores push away.")
        ratings: dict[str, int] = {}
        for label in st.session_state.seen_labels:
            item = label_to_item.get(label)
            if item is None:
                continue
            st.write(label)
            rating = st.radio(
                label,
                options=list(range(1, 11)),
                index=4,
                horizontal=True,
                key=f"rating_{_slug(item['title'])}",
                label_visibility="collapsed",
            )
            ratings[item["title"]] = rating
        st.space("small")
        if st.button(":material/recommend: Get recommendations", type="primary", key="stage2_next"):
            with st.spinner("Applying relevance feedback…"):
                final = apply_user_feedback(
                    original_query_vector=st.session_state.query_vector,
                    rated_items=ratings,
                    candidate_pool=st.session_state.initial_pool,
                    top_k=_TOP_K,
                )
            st.session_state.final_results = final
            _go_to(3)
        col_back, col_reset = st.columns(2)
        with col_back:
            if st.button(":material/arrow_back: Back", key="stage2_back"):
                _go_to(1)
        with col_reset:
            if st.button(":material/restart_alt: Start over", key="stage2_reset"):
                _reset()


def _main_stage_3() -> None:
    had_feedback = bool(st.session_state.seen_labels)
    if had_feedback:
        heading = "Your personalised picks"
        meta = "Re-ranked via Rocchio relevance feedback on your ratings."
        results = st.session_state.final_results
    else:
        heading = "Top recommendations"
        meta = f"Semantic search for \"{_esc(st.session_state.query)}\""
        results = st.session_state.top_k_results

    st.html(_steps_html(active=4))
    st.html(f"""
<div class="scr-results-header">
  <p class="scr-results-title">{_esc(heading)}</p>
  <p class="scr-results-meta">{meta}</p>
</div>
""")

    if not results:
        st.warning("No recommendations found. Try a different query.")
        return

    for rank, item in enumerate(results[:5], start=1):
        _render_horizontal_card(item, rank)

    if len(results) > 5:
        st.markdown("##### More picks")
        for rank, item in enumerate(results[5:], start=6):
            _render_result_card(item, rank)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Smart Content Recommender",
        page_icon=":material/movie:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    _inject_styles()
    
    # התיקון לסליידר ממוקם כאן כדי לדרוס כל עיצוב קודם
    st.markdown("""
        <style>
            div[data-testid="stSlider"], 
            div[data-testid="stSlider"] * {
                direction: ltr !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    _init_state()

    main_fn = {
        0: _main_stage_0,
        1: _main_stage_1,
        2: _main_stage_2,
        3: _main_stage_3,
    }
    main_fn[st.session_state.stage]()


if __name__ == "__main__":
    main()
