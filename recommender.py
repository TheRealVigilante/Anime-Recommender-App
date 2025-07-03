import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv('data/anime.csv')

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assuming df['combined_features'] contains your rich text per anime
corpus = df['combined_features'].fillna('').tolist()

# Generate embeddings for all anime
embeddings = model.encode(corpus, show_progress_bar=True)


def recommend_anime(title, top_n=5):
    title_lower = title.lower()
    matches = df[df['name'].str.lower() == title_lower]
    if matches.empty:
        return []

    idx = matches.index[0]
    query_embedding = embeddings[idx].reshape(1, -1)

    # Compute cosine similarity between query and all embeddings
    sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

    candidates_count = max(50, top_n * 5)
    sim_indices = sim_scores.argsort()[-candidates_count:][::-1]

    recommendations = df.iloc[sim_indices].copy()

    root = title_lower.split(":")[0].strip()
    recommendations = recommendations[~recommendations['name'].str.lower().str.contains(root, na=False)]

    return recommendations.head(top_n).reset_index(drop=True)
