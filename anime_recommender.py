import os
# Must be set before any transformers/sentence_transformers import to prevent
# TensorFlow from being loaded (TF DLL is broken in this environment).
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

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
CSV_PATH   = 'data/anime-dataset-2026.csv'
CACHE_PATH = 'data/embeddings_cache.joblib'
MODEL_PATH = r'model\all-MiniLM-L6-v2'

# ---------------------------------------------------------------------------
# Load and normalise data (CSV only — fast, always at import time)
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df.rename(columns={
    'Name':      'name',
    'Score':     'score',
    'Rank':      'rank',
    'Type':      'type',
    'Genres':    'genres',
    'Synopsis':  'synopsis',
    'Image URL': 'image_url',
    'Studios':   'studios',
    'Source':    'source',
}, inplace=True)

df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)
df['name'] = df['name'].fillna('').astype(str)
df = df[df['name'].str.strip() != ''].reset_index(drop=True)

df['combined_features'] = (
    'Genres: ' + df['genres'].fillna('') + '. '
    'Type: '   + df['type'].fillna('')   + '. '
    'Studio: ' + df['studios'].fillna('') + '. '
    'Source: ' + df['source'].fillna('') + '. '
    + df['synopsis'].fillna('')
)

corpus = df['combined_features'].tolist()

# ---------------------------------------------------------------------------
# load() — heavy work: model init + encode (or cache hit).
# Called once from app.py via @st.cache_resource so it only runs once per
# server process, and Streamlit can show a spinner while it runs.
# ---------------------------------------------------------------------------
def load():
    """Load model and embeddings. Returns (model, embeddings, norm_scores)."""
    csv_mtime = os.path.getmtime(CSV_PATH)

    if os.path.exists(CACHE_PATH):
        _cache = joblib.load(CACHE_PATH)
        if _cache.get('mtime') == csv_mtime:
            embeddings = _cache['embeddings']
            model = SentenceTransformer(MODEL_PATH)
            norm_scores = (df['score'] / 10.0).values
            return model, embeddings, norm_scores

    # Cache miss or stale — encode from scratch.
    model = SentenceTransformer(MODEL_PATH)
    embeddings = model.encode(corpus, show_progress_bar=False, batch_size=64)
    joblib.dump({'embeddings': embeddings, 'mtime': csv_mtime}, CACHE_PATH)
    norm_scores = (df['score'] / 10.0).values
    return model, embeddings, norm_scores


# ---------------------------------------------------------------------------
# recommend_anime
# ---------------------------------------------------------------------------
def recommend_anime(title, top_n=5, min_score=0.0, *, _resources=None):
    """Return (results_df, matched_title) for *title*.

    Pass _resources=(model, embeddings, norm_scores) from app.py to avoid
    reloading on every call. If omitted, load() is called automatically.
    """
    if _resources is None:
        _, embeddings, norm_scores = load()
    else:
        _, embeddings, norm_scores = _resources

    title_lower = title.lower()
    matches = df[df['name'].str.lower() == title_lower]
    matched_title = title

    if matches.empty:
        all_names = df['name'].tolist()
        result = fuzz_process.extractOne(title, all_names, score_cutoff=80)
        if result is None:
            return pd.DataFrame(), None
        matched_title = result[0]
        matches = df[df['name'] == matched_title]

    idx = matches.index[0]
    query_embedding = embeddings[idx].reshape(1, -1)

    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()
    blended = COSINE_WEIGHT * sim_scores + SCORE_WEIGHT * norm_scores

    candidates_count = min(200, len(df))
    sim_indices = blended.argsort()[-candidates_count:][::-1]

    recommendations = df.iloc[sim_indices].copy()
    recommendations['similarity_score'] = blended[sim_indices]
    recommendations = recommendations[recommendations.index != idx]

    # Franchise dedup — exclude sequels, movies, and variants of the queried title.
    # Anime naming has many patterns: "Naruto: Shippuuden", "SAO II", "Dragon Ball Z",
    # "Bleach Movie 1: ...", "Boku no Hero Academia 2nd Season", etc.
    # Strategy: strip the query down to its bare franchise root (drop everything after
    # ':', and trailing season/part/movie/OVA markers), then use rapidfuzz partial_ratio
    # to catch any candidate that contains that root. Threshold of 85 avoids false
    # positives while catching all common sequel/variant patterns.
    import re
    from rapidfuzz.fuzz import partial_ratio

    # Build the bare franchise root from the matched title.
    franchise_root = matched_title.lower()
    # Drop subtitle after ':'
    franchise_root = franchise_root.split(':')[0]
    # Drop trailing season/part/movie/ova/special markers and ordinals
    franchise_root = re.sub(
        r'\s+(?:\d+(?:st|nd|rd|th)?|[ivxlcdm]+|season|part|movie|ova|oad|special|film|arc|hen|final|the)\b.*$',
        '', franchise_root, flags=re.IGNORECASE
    ).strip()

    recommendations = recommendations[
        recommendations['name'].apply(
            lambda n: partial_ratio(franchise_root, str(n).lower()) < 85
        )
    ]

    if min_score > 0.0:
        recommendations = recommendations[recommendations['score'] >= min_score]

    return recommendations.head(top_n).reset_index(drop=True), matched_title
