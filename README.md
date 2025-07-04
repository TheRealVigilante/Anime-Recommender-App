# 🎌 Anime Recommender App

A simple, interactive anime recommendation engine built with **Streamlit** and powered by **Sentence-BERT embeddings** for semantic similarity. Just pick an anime you like, and get personalized, high-quality recommendations instantly!

## 🚀 Features

- 🔍 Choose your favorite anime by directly searching for it from a dropdown list
- 🤖 Recommends similar anime based on synopsis, genre, and more
- 📉 Filter by minimum rating and number of recommendations
- 🖼️ Includes posters, genres, and synopsis previews
- 🔗 Direct links to MyAnimeList (MAL) pages

## 🛠 Installation

### 1. Download the data from the link below
I used my processed version of **Top 15,000 Ranked Anime Dataset** uploaded on HuggingFace

🔗 [Processed HuggingFace Dataset Link](https://huggingface.co/datasets/TheVigilante/Processed_Top_15k_Anime/blob/main/anime.csv)

### 2. Clone the repo

```bash
git clone https://github.com/TheRealVigilante/anime-recommender-app.git
cd anime-recommender-app
````

### 3. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 4. Install requirements

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

It will automatically open [localhost](http://localhost:8501) in your browser.


## 👨‍💻 Author

### Built with passion by [Youssef ElNahas](https://github.com/TheRealVigilante) — aka *TheVigilante* 

If you have any questions, ideas, or feedback — feel free to reach out and connect!
