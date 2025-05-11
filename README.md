# French Lyrics Classification

NLP project for analyzing French song lyrics corpus and predicting metadata like artist, album, genre or year.

## Project Structure

```
.
├── lyrics_dataset/            # Raw data (lyrics with metadata) 
├── tokenized_lyrics_dataset/  # Tokenized data (auto-generated)
├── vectors/                   # Vector representations (auto-generated)
├── utils/                     # Utility modules
│   ├── tokenizer.py           # BPE Tokenizer
│   ├── data_loader.py         # Data loading
│   ├── vectorizers.py         # Vectorization methods
│   └── models.py              # Classification models
└── main.py                    # Main script
```

## Installation

```bash
git clone <repo_url>
cd nlp-project-music
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Usage

```bash
# Full pipeline (tokenize + classify)
python main.py

# Tokenize only
python main.py --mode tokenize

# Classify only
python main.py --mode classify

# Change target to predict
python main.py --label artiste  # Options: artiste, album, genre, année

# Select vectorization methods
python main.py --vectorizers tfidf transformer  # Options: bow, tfidf, word2vec, fasttext, transformer, all

# Save vectors
python main.py --save_vectors
```

## Embedding Methods

1. **Bag-of-Words (BOW)**
2. **TF-IDF**
3. **Word2Vec** (Gensim)
4. **FastText** (Gensim)
5. **Transformer** (`paraphrase-multilingual-MiniLM-L12-v2`)
