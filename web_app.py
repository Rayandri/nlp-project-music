#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface web simple pour tester le classifieur d'artistes
Exécuter avec: python web_app.py
"""

import os
import sys
import pickle
from flask import Flask, request, render_template, redirect, url_for
import numpy as np

# Import des classes pour la prédiction
from utils.models import TextClassifier
from utils.vectorizers import TextVectorizer

app = Flask(__name__)

# Chemin vers le modèle
MODEL_PATH = "results/models/best_artiste.pkl"

def check_model_exists():
    """Vérifie si le modèle existe"""
    return os.path.exists(MODEL_PATH)

def predict_lyrics(text):
    """Prédit l'artiste à partir des paroles"""
    if not check_model_exists():
        return "Modèle non trouvé", []
    
    try:
        # Charger le modèle
        classifier, vectorizer = TextClassifier.load_model(MODEL_PATH)
        
        # Vectoriser le texte
        X = vectorizer.transform([text])
        
        # Prédire l'artiste
        prediction = classifier.predict(X)[0]
        
        # Récupérer les probabilités
        scores = []
        if hasattr(classifier.model, "predict_proba"):
            probs = classifier.model.predict_proba(X)[0]
            classes = classifier.model.classes_
            
            # Trier par probabilité décroissante
            sorted_indices = np.argsort(probs)[::-1]
            scores = [(classes[idx], probs[idx]) for idx in sorted_indices[:5]]
        
        return prediction, scores
    
    except Exception as e:
        return f"Erreur: {str(e)}", []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None
    lyrics = ""
    models_ready = check_model_exists()
    
    if request.method == 'POST':
        lyrics = request.form.get('lyrics', '')
        
        prediction, probabilities = predict_lyrics(lyrics)
    
    return render_template(
        'index.html', 
        lyrics=lyrics,
        prediction=prediction,
        probabilities=probabilities,
        models_ready=models_ready
    )

if __name__ == '__main__':
    # Vérifier si Flask est installé
    try:
        import flask
    except ImportError:
        print("Flask n'est pas installé. Installation en cours...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("Flask installé avec succès.")
    
    print("Serveur démarré: http://127.0.0.1:5000")
    app.run(debug=True) 
