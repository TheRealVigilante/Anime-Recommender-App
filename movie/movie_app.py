import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import streamlit as st
from archive.movie_recommender import recommend_movies, df as movie_df
from archive.movie_recommender import load as load_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Warm cinematic dark gold palette */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0a0a0a 0%, #1a1200 50%, #0a0d1a 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0a0a0a; }

    /* Title */
    h1 { color: #ffd54f !important; letter-spacing: 2px; }
    h3 { color: #ffe082 !important; }

    /* Selectbox / slider labels */
    label, .stSlider label { color: #ffcc02 !important; }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid #5d4037;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary {
        color: #ffe082 !important;
        font-weight: 600;
    }

    /* Recommend button */
    div.stButton > button {
        background: linear-gradient(90deg, #e65100, #bf360c);
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
    p, div, span, li { color: #fff8e1; }
    a { color: #ffd54f !important; }
    hr { border-color: #5d4037; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model + embeddings (cached — runs once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading model and embeddings... (first run only)")
def get_resources():
    return load_movies()

resources = get_resources()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎬 Movie Recommender")
st.caption("Find movies similar to your favourites using semantic AI matching.")
st.markdown("---")

movie_titles = sorted(movie_df['title'].tolist())
selected = st.selectbox("🎯 Choose a movie:", movie_titles)

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("🔢 Recommendations", 3, 15, 5)
with col2:
    min_score = st.slider("⭐ Minimum rating", 0.0, 10.0, 6.5, step=0.5)

st.markdown("")

if st.button("🚀 Find Similar Movies"):
    data = movie_df[movie_df['title'] == selected].iloc[0]

    # --- Selected movie info ---
    img_col, info_col = st.columns([1, 2])
    with img_col:
        poster = data['poster_path']
        if isinstance(poster, str) and poster.strip():
            st.image(f"https://image.tmdb.org/t/p/w300{poster}", width=180)
    with info_col:
        st.markdown(f"## {data['title']}")
        st.markdown(
            f"**Rating:** `{data['vote_average']}/10` &nbsp;({int(data['vote_count'])} votes)&nbsp;|&nbsp;"
            f"**Runtime:** `{data['runtime']} min` &nbsp;|&nbsp;"
            f"**Released:** `{data['release_date']}`"
        )
        st.markdown(f"**Genres:** {data['genres']}")
        imdb = data['imdb_id']
        if isinstance(imdb, str) and imdb.strip():
            st.markdown(f"[🔗 View on IMDB](https://www.imdb.com/title/{imdb}/)")
        if isinstance(data['tagline'], str) and data['tagline'].strip():
            st.markdown(f"*{data['tagline']}*")

    st.markdown(f"> {str(data['overview'])[:500]}...")

    st.markdown("---")
    st.markdown(f"### 🔁 Top {top_n} Similar Movies (rating ≥ {min_score})")

    with st.spinner("Finding recommendations..."):
        recs, matched_title = recommend_movies(
            selected, top_n=top_n, min_score=min_score, _resources=resources
        )

    if matched_title and matched_title != selected:
        st.info(f"Showing results for: *{matched_title}*")

    if len(recs) == 0:
        st.warning("No recommendations found for the current filters. Try lowering the minimum rating.")
    else:
        for _, rec in recs.iterrows():
            pct = rec['similarity_score'] * 100
            with st.expander(f"🎬 {rec['title']}  ·  {rec['vote_average']}/10  ·  {pct:.0f}% match"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    rec_poster = rec['poster_path']
                    if isinstance(rec_poster, str) and rec_poster.strip():
                        st.image(f"https://image.tmdb.org/t/p/w200{rec_poster}", width=120)
                with c2:
                    st.markdown(f"**Genres:** {rec['genres']}")
                    st.markdown(f"**Runtime:** {rec['runtime']} min  &nbsp;|&nbsp;  **Rating:** {rec['vote_average']}/10")
                    rec_imdb = rec['imdb_id']
                    overview_text = str(rec['overview'])[:350] + "..."
                    if isinstance(rec_imdb, str) and rec_imdb.strip():
                        st.write(f"{overview_text} [More Info](https://www.imdb.com/title/{rec_imdb}/)")
                    else:
                        st.write(overview_text)
