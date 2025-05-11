#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

from utils.tokenizer import BPETokenizer
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier

MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

def parse_args():
    parser = argparse.ArgumentParser(description="Prédire l'artiste d'une chanson")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Texte des paroles")
    input_group.add_argument("--file", type=str, help="Fichier contenant les paroles")
    
    parser.add_argument("--train", action="store_true", help="Forcer le réentraînement du modèle")
    return parser.parse_args()

def check_models_exist():
    return (os.path.exists(TOKENIZER_PATH) and 
            os.path.exists(VECTORIZER_PATH) and 
            os.path.exists(CLASSIFIER_PATH) and
            os.path.exists(LABELS_PATH))

def train_models():
    import subprocess
    from pathlib import Path
    
    Path(MODEL_DIR).mkdir(exist_ok=True)
    
    print("Entraînement des modèles...")
    subprocess.run(["python", "main.py", "--mode", "best", "--save_models", MODEL_DIR])

def load_models():
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
        
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(CLASSIFIER_PATH, 'rb') as f:
        classifier = pickle.load(f)
        
    with open(LABELS_PATH, 'rb') as f:
        labels_map = pickle.load(f)
        
    return tokenizer, vectorizer, classifier, labels_map

def predict_artist(text, tokenizer, vectorizer, classifier, labels_map):
    tokenized_text = tokenizer(text)
    X = vectorizer.transform([' '.join(tokenized_text)])
    y_pred = classifier.predict(X)
    
    if hasattr(classifier.model, 'predict_proba'):
        proba = classifier.model.predict_proba(X)[0]
        class_indices = classifier.model.classes_
        
        scores = [(labels_map[idx], prob) for idx, prob in zip(class_indices, proba)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return y_pred[0], scores[:5]
    
    return y_pred[0], []

def main():
    args = parse_args()
    
    if args.text:
        lyrics = args.text
    else:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                lyrics = f.read()
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {e}")
            return 1
    
    if not check_models_exist() or args.train:
        train_models()
    
    try:
        tokenizer, vectorizer, classifier, labels_map = load_models()
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        print("Essayez d'exécuter avec l'option --train pour forcer le réentraînement")
        return 1
    
    try:
        artist, scores = predict_artist(lyrics, tokenizer, vectorizer, classifier, labels_map)
        
        print("\n=== Résultat ===")
        print(f"Artiste prédit: {artist}")
        
        if scores:
            print("\nProbabilités:")
            for name, prob in scores:
                print(f"  {name}: {prob*100:.1f}%")
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
