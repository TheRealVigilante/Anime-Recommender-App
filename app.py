import streamlit as st
import pandas as pd
from recommender import recommend_anime

st.set_page_config(page_title="🎌 Anime Recommender", layout="centered")

df = pd.read_csv('data/anime.csv')  # Ensure it's the cleaned version
anime_titles = sorted(df['name'].tolist())

st.title("🎌 Anime Recommender")

# --- Anime selection ---
selected = st.selectbox("🎯 Choose your favorite anime:", anime_titles)

# --- Controls directly above the button ---
top_n = st.slider("🔢 Number of recommendations", 3, 15, 5)
min_score = st.slider("⭐ Minimum score for recommended anime", 0.0, 10.0, 6.5, step=0.5)

# --- Recommend button ---
if st.button("🚀 Recommend Similar Anime"):
    anime_data = df[df['name'] == selected].iloc[0]

    st.image(anime_data['image_url'], width=200)
    st.markdown(f"**Score:** {anime_data['score']} | **Rank:** #{anime_data['rank']} | **Type:** {anime_data['type']}")
    st.markdown(f"**Genres:** {anime_data['genres']}")
    st.markdown("### 📜 Synopsis")
    st.write(anime_data['synopsis'][:700] + "...")
    st.markdown(f"[🔗 View on MyAnimeList]({anime_data['anime_url']})")  # ← MAL link

    st.markdown("---")
    st.markdown(f"### 🔁 Top {top_n} Recommendations (score ≥ {min_score})")

    recs = recommend_anime(selected, top_n=top_n)
    recs = recs[recs['score'] >= min_score]

    if len(recs) == 0:
        st.warning("No high-quality recommendations found.")
    else:
        for _, rec in recs.iterrows():
            with st.expander(f"🎥 {rec['name']} ({rec['score']}/10)", expanded=False):
                st.image(rec['image_url'], width=150)
                st.markdown(f"**Genres:** {rec['genres']}")
                st.write(rec['synopsis'][:400] + "... [More Info](" + rec['anime_url'] + ")")
