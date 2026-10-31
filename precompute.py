"""
Pre-compute and cache embeddings for both datasets.

Run this once before launching the app:
    conda activate dawn
    python precompute.py

On subsequent app startups the caches are loaded instantly.
Re-run this script whenever you replace a dataset CSV.
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import time
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = r'model\all-MiniLM-L6-v2'

def encode_and_cache(label, csv_path, cache_path, build_corpus_fn):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"  [SKIP] CSV not found: {csv_path}")
        return

    csv_mtime = os.path.getmtime(csv_path)

    if os.path.exists(cache_path):
        cache = joblib.load(cache_path)
        if cache.get('mtime') == csv_mtime:
            print(f"  [OK] Cache is up to date: {cache_path}")
            print(f"       {len(cache['embeddings'])} embeddings already stored.")
            return
        print(f"  [INFO] CSV has changed — recomputing...")
    else:
        print(f"  [INFO] No cache found — computing for the first time...")

    print(f"  Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    corpus = build_corpus_fn(df)
    print(f"  Corpus size: {len(corpus)} entries")

    print(f"  Loading model: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)

    print(f"  Encoding... (this is the slow part)")
    t0 = time.time()
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - t0

    joblib.dump({'embeddings': embeddings, 'mtime': csv_mtime}, cache_path)
    print(f"  [DONE] Encoded {len(embeddings)} entries in {elapsed:.1f}s")
    print(f"  Cache saved to: {cache_path}")


# --- Anime corpus builder ---
def build_anime_corpus(df):
    df = df.rename(columns={
        'Name': 'name', 'Score': 'score', 'Rank': 'rank', 'Type': 'type',
        'Genres': 'genres', 'Synopsis': 'synopsis', 'Image URL': 'image_url',
        'Studios': 'studios', 'Source': 'source',
    })
    return (
        'Genres: ' + df['genres'].fillna('') + '. '
        'Type: '   + df['type'].fillna('')   + '. '
        'Studio: ' + df['studios'].fillna('') + '. '
        'Source: ' + df['source'].fillna('') + '. '
        + df['synopsis'].fillna('')
    ).tolist()


# --- Movie corpus builder ---
def build_movie_corpus(df):
    return (
        'Genres: '   + df['genres'].fillna('')            + '. '
        'Keywords: ' + df['keywords'].fillna('')          + '. '
        'Language: ' + df['original_language'].fillna('') + '. '
        'Tagline: '  + df['tagline'].fillna('')           + '. '
        + df['overview'].fillna('')
    ).tolist()


if __name__ == '__main__':
    encode_and_cache(
        label     = 'Anime dataset  (anime-dataset-2026.csv)',
        csv_path  = 'data/anime-dataset-2026.csv',
        cache_path= 'data/embeddings_cache.joblib',
        build_corpus_fn = build_anime_corpus,
    )
    encode_and_cache(
        label     = 'Movie dataset  (TMDB_movie_dataset_v11.csv)',
        csv_path  = 'data/TMDB_movie_dataset_v11.csv',
        cache_path= 'data/movie_embeddings_cache.joblib',
        build_corpus_fn = build_movie_corpus,
    )
    print("\nAll caches ready. You can now run: streamlit run app.py\n")
