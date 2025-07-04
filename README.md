# 🎌 Anime Recommender App

A simple, interactive anime recommendation engine built with **Streamlit** and powered by **Sentence-BERT embeddings** for semantic similarity. Just pick an anime you like, and get personalized, high-quality recommendations instantly!

## 🚀 Features

- 🔍 Choose your favorite anime by directly searching for it from a dropdown list
- 🤖 Recommends similar anime based on synopsis, genre, and more
- 📉 Filter by minimum rating and number of recommendations
- 🖼️ Includes posters, genres, and synopsis previews
- 🔗 Direct links to MyAnimeList (MAL) pages

## 📦 Dataset

We use the **Top 15,000 Ranked Anime Dataset** from Kaggle

- 🔗 [Unprocessed Kaggle Dataset Link](https://www.kaggle.com/datasets/quanthan/top-15000-ranked-anime-dataset-update-to-32025?select=top_anime_dataset.csv)
- 🔗 [Processed HuggingFace Dataset Link](https://huggingface.co/datasets/TheVigilante/Processed_Top_15k_Anime/blob/main/anime.csv)

---

## 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/TheRealVigilante/anime-recommender-app.git
cd anime-recommender-app
````

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

---

## 📁 Folder Structure

```
.
├── app.py                    # Streamlit UI
├── recommender.py            # Semantic recommendation logic
├── data/
│   └── anime.csv             # Cleaned anime dataset
├── requirements.txt
└── README.md
```

## 🧠 How It Works

* Uses [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) Sentence-BERT to embed anime descriptions and metadata.
* Computes cosine similarity between your chosen anime and others.
* Filters out very similar name roots (e.g., prevents "Naruto" from recommending "Naruto: Shippuden").

---

## ▶️ Running the App

```bash
streamlit run app.py
```

It will automatically open [http://localhost:8501](http://localhost:8501) in your browser.


## 👨‍💻 Author

### Built with passion by [Youssef ElNahas](https://github.com/TheRealVigilante) — aka *TheVigilante* 

If you have any questions, ideas, or feedback — feel free to reach out and connect!
