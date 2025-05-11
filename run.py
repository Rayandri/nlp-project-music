#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script simplifié pour lancer la classification de paroles avec les meilleurs paramètres.
Exécutez simplement: python run.py
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Interface simplifiée pour la classification de paroles")
    parser.add_argument("--dataset", type=str, default="lyrics_dataset",
                       help="Dossier contenant les paroles")
    parser.add_argument("--label", type=str, default="artiste",
                       choices=["artiste", "album", "genre", "année"],
                       help="Type de classification")
    parser.add_argument("--mode", type=str, default="best",
                       choices=["best", "custom"],
                       help="Mode de fonctionnement (best=paramètres optimaux, custom=paramètres personnalisés)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Classification de paroles musicales ===")
    print(f"Dataset: {args.dataset}")
    print(f"Classification par: {args.label}")
    
    if args.mode == "custom":
        # Mode interactif pour paramètres personnalisés
        print("\nMode personnalisé - choisissez vos paramètres:")
        
        min_samples = input("Minimum d'exemples par classe [5]: ").strip() or "5"
        top_classes = input("Nombre de classes à conserver [15]: ").strip() or "15"
        
        print("\nMéthode de vectorisation:")
        print("1. Bag of Words (BOW)")
        print("2. TF-IDF")
        print("3. Word2Vec")
        print("4. FastText")
        print("5. Transformer")
        vectorizer_choice = input("Choix [1]: ").strip() or "1"
        
        vectorizers = {
            "1": "bow",
            "2": "tfidf",
            "3": "word2vec",
            "4": "fasttext",
            "5": "transformer"
        }.get(vectorizer_choice, "bow")
        
        print("\nClassifieur:")
        print("1. Régression logistique")
        print("2. SVM")
        print("3. Random Forest")
        classifier_choice = input("Choix [1]: ").strip() or "1"
        
        classifiers = {
            "1": "logistic",
            "2": "svm",
            "3": "random_forest"
        }.get(classifier_choice, "logistic")
        
        use_stopwords = input("Filtrer les mots vides? (o/n) [o]: ").strip().lower() or "o"
        use_stopwords = use_stopwords.startswith("o")
        
        bpe_merges = input("Nombre de fusions BPE [1500]: ").strip() or "1500"
        
        # Construire la commande avec les paramètres personnalisés
        cmd = [
            "python", "main.py",
            "--mode", "all",
            "--label", args.label,
            "--min_samples", min_samples,
            "--top_classes", top_classes,
            "--vectorizers", vectorizers,
            "--classifier", classifiers,
            "--bpe_merges", bpe_merges
        ]
        
        if use_stopwords:
            cmd.append("--use_stopwords")
        
        print("\nExécution de la commande:")
        print(" ".join(cmd))
        
    else:
        # Mode automatique avec les meilleurs paramètres
        cmd = [
            "python", "main.py",
            "--mode", "best",
            "--label", args.label
        ]
        print("\nUtilisation des paramètres optimaux")
    
    # Exécuter la commande
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
