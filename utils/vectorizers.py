import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer

class TextVectorizer:
    """
    Text vectorization with multiple methods
    """
    def __init__(self, method: str = "tfidf", vector_size: int = 100, **kwargs):
        """
        Initialize vectorizer
        
        Args:
            method: Vectorization method ('bow', 'tfidf', 'word2vec', 'fasttext', 'transformer')
            vector_size: Size of Word2Vec and FastText vectors
            **kwargs: Additional model parameters
        """
        self.method = method.lower()
        self.vector_size = vector_size
        self.model = None
        self.kwargs = kwargs
        
        # Initialize model based on method
        if self.method == "bow":
            self.model = CountVectorizer(**kwargs)
        elif self.method == "tfidf":
            self.model = TfidfVectorizer(**kwargs)
        elif self.method == "transformer":
            model_name = kwargs.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
            self.model = SentenceTransformer(model_name)
    
    def fit(self, documents: List[str]) -> None:
        """
        Train vectorizer on documents
        
        Args:
            documents: List of document texts
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
        Transform documents to vectors
        
        Args:
            documents: List of document texts
            
        Returns:
            Matrix of vectorized documents
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
        """
        Train vectorizer and transform documents
        
        Args:
            documents: List of document texts
            
        Returns:
            Matrix of vectorized documents
        """
        self.fit(documents)
        return self.transform(documents)
    
    def _document_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Calculate average document vector for Word2Vec and FastText
        
        Args:
            tokens: List of document tokens
            
        Returns:
            Average document vector
        """
        if self.method == "word2vec":
            valid_tokens = [token for token in tokens if token in self.model.wv.key_to_index]
        elif self.method == "fasttext":
            valid_tokens = tokens  # FastText can handle unknown words
        else:
            return np.zeros(self.vector_size)
            
        if not valid_tokens:
            return np.zeros(self.vector_size)
            
        if self.method == "word2vec":
            return np.mean(self.model.wv[valid_tokens], axis=0)
        elif self.method == "fasttext":
            return np.mean([self.model.wv[token] for token in valid_tokens], axis=0) 
