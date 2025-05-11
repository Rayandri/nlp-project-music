import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

class TextVectorizer:
    """
    Classe pour vectoriser les textes selon différentes méthodes
    """
    def __init__(self, method: str = "tfidf", vector_size: int = 100, **kwargs):
        """
        Initialise le vectoriseur
        
        Args:
            method: Méthode de vectorisation ('bow', 'tfidf', 'word2vec', 'fasttext', 'transformer')
            vector_size: Taille des vecteurs Word2Vec et FastText
            **kwargs: Arguments supplémentaires pour les modèles
        """
        self.method = method.lower()
        self.vector_size = vector_size
        self.model = None
        self.kwargs = kwargs
        
        # Initialiser le modèle selon la méthode choisie
        if self.method == "bow":
            self.model = CountVectorizer(**kwargs)
        elif self.method == "tfidf":
            self.model = TfidfVectorizer(**kwargs)
        elif self.method == "transformer":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
            self.model = SentenceTransformer(model_name)
    
    def fit(self, documents: List[str]) -> None:
        """
        Entraîne le vectoriseur sur les documents
        
        Args:
            documents: Liste de documents (textes)
        """
        if self.method in ["bow", "tfidf"]:
            self.model.fit(documents)
        elif self.method == "word2vec":
            tokenized_docs = [doc.split() for doc in documents]
            self.model = Word2Vec(
                sentences=tokenized_docs, 
                vector_size=self.vector_size, 
                window=self.kwargs.get("window", 5), 
                min_count=self.kwargs.get("min_count", 2), 
                workers=self.kwargs.get("workers", 4), 
                epochs=self.kwargs.get("epochs", 10)
            )
        elif self.method == "fasttext":
            tokenized_docs = [doc.split() for doc in documents]
            self.model = FastText(
                sentences=tokenized_docs, 
                vector_size=self.vector_size, 
                window=self.kwargs.get("window", 5), 
                min_count=self.kwargs.get("min_count", 2), 
                workers=self.kwargs.get("workers", 4), 
                epochs=self.kwargs.get("epochs", 10)
            )
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforme les documents en vecteurs
        
        Args:
            documents: Liste de documents (textes)
            
        Returns:
            Matrice des documents vectorisés
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
            
        if self.method in ["bow", "tfidf"]:
            X = self.model.transform(documents)
            return X.toarray()
        elif self.method == "transformer":
            return self.model.encode(documents, show_progress_bar=True)
        elif self.method in ["word2vec", "fasttext"]:
            tokenized_docs = [doc.split() for doc in documents]
            return np.array([self._document_vector(doc) for doc in tokenized_docs])
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Entraîne le vectoriseur et transforme les documents
        
        Args:
            documents: Liste de documents (textes)
            
        Returns:
            Matrice des documents vectorisés
        """
        self.fit(documents)
        return self.transform(documents)
    
    def _document_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Calcule le vecteur moyen d'un document pour Word2Vec et FastText
        
        Args:
            tokens: Liste de tokens d'un document
            
        Returns:
            Vecteur moyen du document
        """
        if self.method == "word2vec":
            valid_tokens = [token for token in tokens if token in self.model.wv.key_to_index]
        elif self.method == "fasttext":
            valid_tokens = tokens  # FastText peut gérer les mots inconnus
        else:
            return np.zeros(self.vector_size)
            
        if not valid_tokens:
            return np.zeros(self.vector_size)
            
        if self.method == "word2vec":
            return np.mean(self.model.wv[valid_tokens], axis=0)
        elif self.method == "fasttext":
            return np.mean([self.model.wv[token] for token in valid_tokens], axis=0) 
