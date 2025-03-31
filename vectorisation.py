import os
import glob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

def load_documents(root_dir):
    documents = []
    file_paths = []
    # Parcours récursif de tous les fichiers .txt dans tokenized_lyrics_dataset
    for filepath in glob.glob(os.path.join(root_dir, '**', '*.txt'), recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:  # éviter les documents vides
                documents.append(text)
                file_paths.append(filepath)
    return documents, file_paths

# Chargement des textes tokenisés
input_dir = "tokenized_lyrics_dataset"
documents, file_paths = load_documents(input_dir)
print(f"{len(documents)} documents chargés.")

# --- 1. Vectorisation Bag-of-Words (BOW) ---
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)
print("BOW shape:", bow_matrix.shape)

# --- 2. Vectorisation TF-IDF ---
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF shape:", tfidf_matrix.shape)

# --- Préparation du corpus tokenisé pour word embeddings ---
# Puisque nos fichiers sont déjà tokenisés (tokens séparés par des espaces),
# on peut simplement splitter par espace.
tokenized_docs = [doc.split() for doc in documents]

# --- 3. Word2Vec ---
# Entraînement d'un modèle Word2Vec sur le corpus tokenisé
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

# Fonction pour calculer l'embedding moyen d'un document à partir de Word2Vec
def document_vector_w2v(tokens, model):
    # Filtrer les tokens non présents dans le vocabulaire
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

w2v_embeddings = np.array([document_vector_w2v(doc, w2v_model) for doc in tokenized_docs])
print("Word2Vec embeddings shape:", w2v_embeddings.shape)

# --- 4. FastText ---
# Entraînement d'un modèle FastText sur le même corpus
ft_model = FastText(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

def document_vector_ft(tokens, model):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

ft_embeddings = np.array([document_vector_ft(doc, ft_model) for doc in tokenized_docs])
print("FastText embeddings shape:", ft_embeddings.shape)

# --- 5. Embeddings contextuels avec Transformers ---
transformer_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
transformer_embeddings = transformer_model.encode(documents, show_progress_bar=True)
print("Transformer embeddings shape:", transformer_embeddings.shape)

np.save("bow_matrix.npy", bow_matrix.toarray())
np.save("tfidf_matrix.npy", tfidf_matrix.toarray())
np.save("w2v_embeddings.npy", w2v_embeddings)
np.save("ft_embeddings.npy", ft_embeddings)
np.save("transformer_embeddings.npy", transformer_embeddings)
