import os
import glob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

# --- Chargement des documents tokenisés ---
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

tokenized_root = "tokenized_lyrics_dataset"
documents, file_paths = load_documents(tokenized_root)
print(f"{len(documents)} documents chargés.")

# --- Extraction des labels (artiste) ---
def extract_artist_label(original_filepath):
    """
    Lit le fichier original (non tokenisé) et extrait la valeur associée à la ligne commençant par "Artiste :"
    On suppose que le header se trouve en début de fichier et est séparé d'une ligne vide.
    """
    try:
        with open(original_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # On parcourt les lignes jusqu'à la première ligne vide (fin du header)
        for line in lines:
            line = line.strip()
            if not line:
                break
            # Si la ligne commence par "Artiste" (insensible à la casse)
            if line.lower().startswith("artiste"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
    except Exception as e:
        print("Erreur pour", original_filepath, e)
    return None

# Pour chaque fichier tokenisé, on déduit le chemin du fichier original en remplaçant le dossier racine
original_root = "lyrics_dataset"
labels = []
valid_idx = []  # index des fichiers pour lesquels on a trouvé une étiquette

for i, tokenized_path in enumerate(file_paths):
    # Calculer le chemin relatif et reconstruire le chemin dans lyrics_dataset
    rel_path = os.path.relpath(tokenized_path, tokenized_root)
    original_path = os.path.join(original_root, rel_path)
    artist = extract_artist_label(original_path)
    if artist is not None:
        labels.append(artist)
        valid_idx.append(i)
    else:
        labels.append("Unknown")  # ou on peut ignorer ce document

# Filtrer documents, file_paths et labels selon valid_idx
documents = [documents[i] for i in valid_idx]
file_paths = [file_paths[i] for i in valid_idx]
labels = [labels[i] for i in valid_idx]
print(f"{len(documents)} documents avec label 'Artiste' extraits.")

# --- Préparation des représentations vectorielles ---

# 1. Bag-of-Words (BOW)
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)
print("BOW shape:", bow_matrix.shape)

# 2. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF shape:", tfidf_matrix.shape)

# Préparation pour les embeddings : les fichiers tokenisés contiennent des tokens séparés par espaces.
tokenized_docs = [doc.split() for doc in documents]

# 3. Word2Vec
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

def document_vector(model, tokens):
    valid_tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

w2v_embeddings = np.array([document_vector(w2v_model, doc) for doc in tokenized_docs])
print("Word2Vec embeddings shape:", w2v_embeddings.shape)

# 4. FastText
ft_model = FastText(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
ft_embeddings = np.array([document_vector(ft_model, doc) for doc in tokenized_docs])
print("FastText embeddings shape:", ft_embeddings.shape)

# 5. Embeddings contextuels avec Transformers
transformer_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
transformer_embeddings = transformer_model.encode(documents, show_progress_bar=True)
print("Transformer embeddings shape:", transformer_embeddings.shape)

# --- Entraînement de classifieurs pour prédire l'artiste à partir des représentations ---
from sklearn.linear_model import LogisticRegression

# Fonction d'entraînement et d'évaluation
def train_and_evaluate(X, y, representation_name):
    # On convertit X en matrice dense si nécessaire (pour BOW et TF-IDF)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[{representation_name}] Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- Classification avec différentes représentations ---")
train_and_evaluate(bow_matrix, labels, "BOW")
train_and_evaluate(tfidf_matrix, labels, "TF-IDF")
train_and_evaluate(w2v_embeddings, labels, "Word2Vec")
train_and_evaluate(ft_embeddings, labels, "FastText")
train_and_evaluate(transformer_embeddings, labels, "Transformer")
