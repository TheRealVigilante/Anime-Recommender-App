import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as fuzz_process

# --- Blend weights ---
COSINE_WEIGHT = 0.7
SCORE_WEIGHT  = 0.3

# --- Paths ---
CSV_PATH   = 'data/TMDB_movie_dataset_v11.csv'
CACHE_PATH = 'data/movie_embeddings_cache.joblib'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Coerce vote_average to float and fill missing with 0.0.
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0)

# ---------------------------------------------------------------------------
# Enriched combined_features
# Columns used: genres, keywords, original_language, tagline, overview
# ---------------------------------------------------------------------------
df['combined_features'] = (
    'Genres: '   + df['genres'].fillna('')            + '. '
    'Keywords: ' + df['keywords'].fillna('')          + '. '
    'Language: ' + df['original_language'].fillna('') + '. '
    'Tagline: '  + df['tagline'].fillna('')            + '. '
    + df['overview'].fillna('')
)

corpus = df['combined_features'].tolist()

# ---------------------------------------------------------------------------
# Cache embeddings with joblib
# Cache key: CSV file mtime; recompute if CSV has changed.
# ---------------------------------------------------------------------------
model = SentenceTransformer(r'model\all-MiniLM-L6-v2')

csv_mtime = os.path.getmtime(CSV_PATH)

if os.path.exists(CACHE_PATH):
    _cache = joblib.load(CACHE_PATH)
    if _cache.get('mtime') == csv_mtime:
        embeddings = _cache['embeddings']
    else:
        embeddings = model.encode(corpus, show_progress_bar=True)
        joblib.dump({'embeddings': embeddings, 'mtime': csv_mtime}, CACHE_PATH)
else:
    embeddings = model.encode(corpus, show_progress_bar=True)
    joblib.dump({'embeddings': embeddings, 'mtime': csv_mtime}, CACHE_PATH)

# Pre-compute normalised scores once at module level for fast blending.
_norm_scores = (df['vote_average'] / 10.0).values


# ---------------------------------------------------------------------------
# recommend_movies
# ---------------------------------------------------------------------------
def recommend_movies(title, top_n=5, min_score=0.0):
    """Return (results_df, matched_title) for *title*.

    - Fuzzy title matching via rapidfuzz (cutoff=80).
      Always returns a tuple; matched_title may differ from *title* on a
      fuzzy hit, and is None on complete failure.
    - min_score filter applied inside the function after over-fetching.
    - Blended cosine+score ranking.
    - 'similarity_score' column on the returned DataFrame.
    - Prefix-only dedup: exclude results whose title starts with the same
      root (before ':') as the query.
    """
    title_lower = title.lower()
    matches = df[df['title'].str.lower() == title_lower]
    matched_title = title  # updated to canonical title if fuzzy matching fires

    if matches.empty:
        # Fuzzy match with cutoff=80 (score out of 100)
        all_titles = df['title'].tolist()
        result = fuzz_process.extractOne(title, all_titles, score_cutoff=80)
        if result is None:
            return pd.DataFrame(), None
        matched_title = result[0]
        matches = df[df['title'] == matched_title]

    idx = matches.index[0]
    query_embedding = embeddings[idx].reshape(1, -1)

    # Cosine similarity against all entries.
    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Blend cosine similarity with normalised score.
    blended = COSINE_WEIGHT * sim_scores + SCORE_WEIGHT * _norm_scores

    # Over-fetch up to 200 candidates before filtering.
    candidates_count = min(200, len(df))
    sim_indices = blended.argsort()[-candidates_count:][::-1]

    recommendations = df.iloc[sim_indices].copy()

    # Attach blended score as similarity_score column.
    recommendations['similarity_score'] = blended[sim_indices]

    # Explicitly remove the queried movie by index so self-recommendation
    # never appears, regardless of whether prefix dedup below catches it.
    recommendations = recommendations[recommendations.index != idx]

    # Prefix-only dedup: exclude results whose title starts with the root
    # word before ':' of the matched title.
    root = matched_title.lower().split(':')[0].strip()
    recommendations = recommendations[
        ~recommendations['title'].str.lower().str.startswith(root, na=False)
    ]

    # Apply vote_average filter inside the function.
    if min_score > 0.0:
        recommendations = recommendations[recommendations['vote_average'] >= min_score]

    return recommendations.head(top_n).reset_index(drop=True), matched_title
