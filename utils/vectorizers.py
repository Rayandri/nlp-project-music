"""
Text vectorization utilities supporting various embedding methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

class TextVectorizer:
    """Text vectorizer supporting multiple methods: BoW, TF-IDF, Word2Vec, FastText, and Transformers."""
    
    def __init__(self, method: str = "tfidf", vector_size: int = 100, **kwargs):
        """Initialize a text vectorizer with the specified method.
        
        Args:
            method: Vectorization method ("bow", "tfidf", "word2vec", "fasttext", "transformer")
            vector_size: Size of output vectors for embedding methods
            **kwargs: Additional parameters specific to each method
        """
        self.method = method.lower()
        self.vector_size = vector_size
        self.model = None
        self.kwargs = kwargs
        
        if self.method == "bow":
            self.model = CountVectorizer(**kwargs)
        elif self.method == "tfidf":
            self.model = TfidfVectorizer(**kwargs)
        elif self.method == "transformer":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
            self.model = SentenceTransformer(model_name)
    
    def fit(self, documents: List[str]) -> None:
        """Fit the vectorizer on the provided documents.
        
        Args:
            documents: List of text documents
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
        """Transform the documents into feature vectors.
        
        Args:
            documents: List of text documents
            
        Returns:
            NumPy array of feature vectors
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        if self.method in ["bow", "tfidf"]:
            X = self.model.transform(documents)
            return X.toarray()
        elif self.method == "transformer":
            return self.model.encode(documents, show_progress_bar=True)
        elif self.method in ["word2vec", "fasttext"]:
            tokenized_docs = [doc.split() for doc in documents]
            return np.array([self._document_vector(doc) for doc in tokenized_docs])
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform the documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            NumPy array of feature vectors
        """
        self.fit(documents)
        return self.transform(documents)
    
    def get_feature_names_out(self) -> List[str]:
        """Return feature names for the vectorizer.
        
        Returns:
            List of feature names
        """
        if self.method in ["bow", "tfidf"]:
            return self.model.get_feature_names_out()
        elif self.method in ["word2vec", "fasttext"]:
            return [f"dim_{i}" for i in range(self.vector_size)]
        elif self.method == "transformer":
            dummy_text = "dummy text for encoding"
            embeddings = self.model.encode([dummy_text])[0]
            return [f"dim_{i}" for i in range(len(embeddings))]
        else:
            return []
    
    def _document_vector(self, tokens: List[str]) -> np.ndarray:
        """Create a document vector by averaging token embeddings.
        
        Args:
            tokens: List of tokens in the document
            
        Returns:
            Document embedding vector
        """
        if self.method == "word2vec":
            valid_tokens = [token for token in tokens if token in self.model.wv.key_to_index]
        elif self.method == "fasttext":
            valid_tokens = tokens
        else:
            return np.zeros(self.vector_size)
            
        if not valid_tokens:
            return np.zeros(self.vector_size)
            
        if self.method == "word2vec":
            return np.mean(self.model.wv[valid_tokens], axis=0)
        elif self.method == "fasttext":
            return np.mean([self.model.wv[token] for token in valid_tokens], axis=0) 
