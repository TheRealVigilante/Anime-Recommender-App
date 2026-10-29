import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import process as fuzz_process

# --- Blend weights (Upgrade 4) ---
COSINE_WEIGHT = 0.7
SCORE_WEIGHT  = 0.3

# --- Paths ---
CSV_PATH   = 'data/anime-dataset-2026.csv'
CACHE_PATH = 'data/embeddings_cache.joblib'

# ---------------------------------------------------------------------------
# Load and normalise data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Rename Title-case CSV columns to lowercase snake_case expected by all downstream code.
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

# Coerce score to float (may be 'UNKNOWN' string) and fill missing with 0.0.
df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)

# ---------------------------------------------------------------------------
# Upgrade 3 — Enrich combined_features with structured metadata
# Columns used: genres, type, studios, source, synopsis
# (No themes/franchise columns exist in anime-dataset-2026.csv.)
# ---------------------------------------------------------------------------
df['combined_features'] = (
    'Genres: ' + df['genres'].fillna('') + '. '
    'Type: '   + df['type'].fillna('')   + '. '
    'Studio: ' + df['studios'].fillna('') + '. '
    'Source: ' + df['source'].fillna('') + '. '
    + df['synopsis'].fillna('')
)

corpus = df['combined_features'].tolist()

# ---------------------------------------------------------------------------
# Upgrade 1 — Cache embeddings with joblib
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
_norm_scores = (df['score'] / 10.0).values


# ---------------------------------------------------------------------------
# recommend_anime — Upgrades 2, 4, 5, 6, 7 all live here
# ---------------------------------------------------------------------------
def recommend_anime(title, top_n=5, min_score=0.0):
    """Return (results_df, matched_title) for *title*.

    - Upgrade 6: fuzzy title matching via rapidfuzz (cutoff=80).
      Always returns a tuple; matched_title may differ from *title* on a
      fuzzy hit, and is None on complete failure.
    - Upgrade 2: min_score filter applied inside the function after over-fetching.
    - Upgrade 4: blended cosine+score ranking.
    - Upgrade 5: 'similarity_score' column on the returned DataFrame.
    - Upgrade 7: prefix-only dedup (no franchise column exists in the CSV).
    """
    title_lower = title.lower()
    matches = df[df['name'].str.lower() == title_lower]
    matched_title = title  # updated to the canonical title if fuzzy matching fires

    if matches.empty:
        # Upgrade 6 — fuzzy match with cutoff=80 (score out of 100)
        all_names = df['name'].tolist()
        result = fuzz_process.extractOne(title, all_names, score_cutoff=80)
        if result is None:
            return pd.DataFrame(), None
        matched_title = result[0]
        matches = df[df['name'] == matched_title]

    idx = matches.index[0]
    query_embedding = embeddings[idx].reshape(1, -1)

    # Cosine similarity against all entries.
    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

    # Upgrade 4 — blend cosine similarity with normalised score.
    blended = COSINE_WEIGHT * sim_scores + SCORE_WEIGHT * _norm_scores

    # Upgrade 2 — over-fetch up to 200 candidates before filtering.
    candidates_count = min(200, len(df))
    sim_indices = blended.argsort()[-candidates_count:][::-1]

    recommendations = df.iloc[sim_indices].copy()

    # Upgrade 5 — attach blended score as similarity_score column.
    # Assign before any row-dropping so the positional alignment is intact.
    recommendations['similarity_score'] = blended[sim_indices]

    # Explicitly remove the queried anime by index so self-recommendation never
    # appears, regardless of whether the prefix dedup below catches it.
    recommendations = recommendations[recommendations.index != idx]

    # Upgrade 7 — prefix-only dedup to avoid over-filtering unrelated titles.
    # (No franchise/series/related column exists in anime-dataset-2026.csv;
    # we exclude only results whose name starts with the root word before ':'.)
    root = matched_title.lower().split(':')[0].strip()
    recommendations = recommendations[
        ~recommendations['name'].str.lower().str.startswith(root, na=False)
    ]

    # Upgrade 2 — apply score filter inside the function.
    if min_score > 0.0:
        recommendations = recommendations[recommendations['score'] >= min_score]

    return recommendations.head(top_n).reset_index(drop=True), matched_title
