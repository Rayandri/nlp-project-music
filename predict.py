#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour prédire le label d'un texte (artiste, genre, etc.)
"""

import os
import sys
import argparse
import numpy as np
from utils.models import TextClassifier
from utils.vectorizers import TextVectorizer

def parse_args():
    parser = argparse.ArgumentParser(description="Prédiction de labels pour les paroles de chansons")
    
    # Source du texte
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Texte à classifier")
    input_group.add_argument("--file", type=str, help="Fichier texte à classifier")
    
    # Paramètres du modèle
    parser.add_argument("--model", type=str, default="results/models/best_artiste.pkl", 
                       help="Chemin vers le modèle préentrainé (fichier .pkl)")
    parser.add_argument("--label", type=str, 
                       choices=["artiste", "album", "genre", "année"], 
                       default="artiste",
                       help="Type de label à prédire")
    
    return parser.parse_args()

def load_text(args):
    """Charge le texte à classifier"""
    if args.text:
        return args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {str(e)}")
            sys.exit(1)
    return None

def main():
    args = parse_args()
    
    # Charger le texte
    text = load_text(args)
    if not text:
        print("Erreur: Aucun texte à classifier")
        return
    
    print(f"\n=== Classification de paroles ({args.label}) ===")
    
    # Vérifier si le modèle existe
    if not os.path.exists(args.model):
        print(f"Erreur: Le modèle {args.model} n'existe pas.")
        return
    
    # Charger le modèle pré-entraîné
    try:
        print(f"Chargement du modèle: {args.model}")
        classifier, vectorizer = TextClassifier.load_model(args.model)
        
        # Si le vectoriseur n'est pas inclus dans le fichier, créer un nouveau
        if vectorizer is None:
            print("Attention: Vectoriseur non trouvé dans le fichier modèle.")
            print("Création d'un vectoriseur TF-IDF par défaut.")
            vectorizer = TextVectorizer(method="tfidf")
            # Dans ce cas, il faudrait réentraîner le vectoriseur sur les données d'origine
            # ce qui n'est pas possible ici, donc les résultats seront probablement mauvais
    
        # Vectoriser le texte
        X = vectorizer.transform([text])
        
        # Prédire le label
        prediction = classifier.predict(X)[0]
        
        # Afficher la prédiction
        print(f"\nTexte analysé: {text[:100]}...")
        print(f"\nPrédiction: {prediction}")
        
        # Si le modèle a des probabilités, les afficher
        if hasattr(classifier.model, "predict_proba"):
            try:
                probs = classifier.model.predict_proba(X)[0]
                classes = classifier.model.classes_
                
                # Trier par probabilité décroissante
                sorted_indices = np.argsort(probs)[::-1]
                
                # Format joli pour l'affichage
                print("\nTop 5 probabilités:")
                for i in range(min(5, len(classes))):
                    idx = sorted_indices[i]
                    print(f"  {classes[idx]}: {probs[idx]:.3f} ({probs[idx]*100:.1f}%)")
                
                # Afficher un résumé coloré (ASCII art simple)
                print("\nRésultat final:")
                prediction_prob = probs[list(classes).index(prediction)]
                confidence = "Haute" if prediction_prob > 0.6 else "Moyenne" if prediction_prob > 0.3 else "Basse"
                print("┌" + "─" * 50 + "┐")
                print(f"│ {args.label.upper()}: {prediction:<43} │")
                print(f"│ Confiance: {confidence:<40} │")
                print(f"│ Probabilité: {prediction_prob*100:.1f}%{' ':<35} │")
                print("└" + "─" * 50 + "┘")
            except Exception as e:
                print(f"Erreur lors du calcul des probabilités: {str(e)}")
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    main() 
