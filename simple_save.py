#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script to generate pickle files with model and vectorizer.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    """Create test models and save them to pickle files."""
    print("Création de modèles de test...")
    os.makedirs("results/models", exist_ok=True)
    texts = [
        "Ceci est un exemple de texte",
        "Voici un autre exemple",
        "Les paroles de chanson sont intéressantes",
        "J'aime la musique rap",
        "Le rock est un style musical"
    ]
    labels = ["classe1", "classe2", "classe1", "classe2", "classe1"]
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    data = {
        'model': model,
        'model_type': 'logistic',
        'vectorizer': vectorizer
    }
    model_path = "results/models/test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Modèle sauvegardé dans: {model_path}")
    if os.path.exists(model_path):
        print(f"Vérification : le fichier {model_path} existe bien")
        print(f"Taille du fichier: {os.path.getsize(model_path)} bytes")
    else:
        print(f"ERREUR: Le fichier {model_path} n'a pas été créé")

if __name__ == "__main__":
    main() 
