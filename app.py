import streamlit as st
from recommender import recommend_anime, df

st.set_page_config(page_title="🎌 Anime Recommender", layout="centered")

# df is already loaded and normalised by recommender.py — import directly
# to avoid reading and renaming the CSV a second time.
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
    st.write(str(anime_data['synopsis'])[:700] + "...")
    # No anime_url column in the CSV; construct MAL link from anime_id.
    mal_url = f"https://myanimelist.net/anime/{anime_data['anime_id']}"
    st.markdown(f"[🔗 View on MyAnimeList]({mal_url})")

    st.markdown("---")
    st.markdown(f"### 🔁 Top {top_n} Recommendations (score ≥ {min_score})")

    # Upgrade 6 — unpack tuple; min_score filtering is done inside the function.
    recs, matched_title = recommend_anime(selected, top_n=top_n, min_score=min_score)

    # Show a note if a fuzzy match was used instead of an exact one.
    if matched_title and matched_title != selected:
        st.info(f"Showing results for: *{matched_title}*")

    if len(recs) == 0:
        st.warning("No high-quality recommendations found.")
    else:
        for _, rec in recs.iterrows():
            # Upgrade 5 — show blended similarity percentage in expander header.
            with st.expander(
                f"🎥 {rec['name']} ({rec['score']}/10) — {rec['similarity_score'] * 100:.0f}% match",
                expanded=False,
            ):
                st.image(rec['image_url'], width=150)
                st.markdown(f"**Genres:** {rec['genres']}")
                rec_mal_url = f"https://myanimelist.net/anime/{rec['anime_id']}"
                st.write(str(rec['synopsis'])[:400] + f"... [More Info]({rec_mal_url})")
