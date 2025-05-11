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

# Import des fonctions de predict.py
from predict import load_models, predict_artist, check_models_exist, train_models

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None
    lyrics = ""
    models_ready = check_models_exist()
    
    if request.method == 'POST':
        lyrics = request.form.get('lyrics', '')
        
        if not models_ready:
            train_models()
        
        try:
            tokenizer, vectorizer, classifier, labels_map = load_models()
            artist, scores = predict_artist(lyrics, tokenizer, vectorizer, classifier, labels_map)
            prediction = artist
            probabilities = scores
        except Exception as e:
            prediction = f"Erreur: {str(e)}"
    
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
