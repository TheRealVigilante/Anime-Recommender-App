import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import streamlit as st
from anime_recommender import recommend_anime, df as anime_df
from anime_recommender import load as load_anime

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Deep navy/purple anime-inspired palette */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0d0d1a 0%, #1a0a2e 50%, #0d1a2e 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0d0d1a; }

    /* Title */
    h1 { color: #e040fb !important; letter-spacing: 2px; }
    h3 { color: #b39ddb !important; }

    /* Selectbox / slider labels */
    label, .stSlider label { color: #ce93d8 !important; }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid #4a148c;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary {
        color: #e1bee7 !important;
        font-weight: 600;
    }

    /* Recommend button */
    div.stButton > button {
        background: linear-gradient(90deg, #7b1fa2, #4a148c);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.85; }

    /* General text */
    p, div, span, li { color: #e1bee7; }
    a { color: #ce93d8 !important; }
    hr { border-color: #4a148c; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model + embeddings (cached — runs once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading model and embeddings... (first run only)")
def get_resources():
    return load_anime()

resources = get_resources()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎌 Anime Recommender")
st.caption("Find anime similar to your favourites using semantic AI matching.")
st.markdown("---")

anime_titles = sorted(anime_df['name'].tolist())
selected = st.selectbox("🎯 Choose an anime:", anime_titles)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("🔢 Recommendations", 3, 15, 5)
with col2:
    min_score = st.slider("⭐ Minimum score", 0.0, 10.0, 6.5, step=0.5)

st.markdown("")

if st.button("🚀 Find Similar Anime"):
    data = anime_df[anime_df['name'] == selected].iloc[0]

    # --- Selected anime info ---
    img_col, info_col = st.columns([1, 2])
    with img_col:
        st.image(data['image_url'], width=180)
    with info_col:
        st.markdown(f"## {data['name']}")
        st.markdown(f"**Score:** `{data['score']}` &nbsp;|&nbsp; **Rank:** `#{data['rank']}` &nbsp;|&nbsp; **Type:** `{data['type']}`")
        st.markdown(f"**Genres:** {data['genres']}")
        mal_url = f"https://myanimelist.net/anime/{data['anime_id']}"
        st.markdown(f"[🔗 View on MyAnimeList]({mal_url})")
    st.markdown(f"> {str(data['synopsis'])[:500]}...")

    st.markdown("---")
    st.markdown(f"### 🔁 Top {top_n} Similar Anime (score ≥ {min_score})")

    with st.spinner("Finding recommendations..."):
        recs, matched_title = recommend_anime(
            selected, top_n=top_n, min_score=min_score, _resources=resources
        )

    if matched_title and matched_title != selected:
        st.info(f"Showing results for: *{matched_title}*")

    if len(recs) == 0:
        st.warning("No recommendations found for the current filters. Try lowering the minimum score.")
    else:
        for _, rec in recs.iterrows():
            pct = rec['similarity_score'] * 100
            with st.expander(f"🎥 {rec['name']}  ·  {rec['score']}/10  ·  {pct:.0f}% match"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(rec['image_url'], width=120)
                with c2:
                    st.markdown(f"**Genres:** {rec['genres']}")
                    st.markdown(f"**Type:** {rec['type']}  &nbsp;|&nbsp;  **Score:** {rec['score']}")
                    rec_mal_url = f"https://myanimelist.net/anime/{rec['anime_id']}"
                    st.write(str(rec['synopsis'])[:350] + f"... [More Info]({rec_mal_url})")
