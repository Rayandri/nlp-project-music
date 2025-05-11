#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script simplifié pour lancer la classification de paroles avec les meilleurs paramètres.
Exécutez simplement: python run.py
"""

import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Interface pour la classification de paroles")
    parser.add_argument("--dataset", type=str, default="lyrics_dataset")
    parser.add_argument("--label", type=str, default="artiste",
                       choices=["artiste", "album", "genre", "année"])
    parser.add_argument("--mode", type=str, default="best",
                       choices=["best", "custom"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== Classification de paroles ===")
    print(f"Dataset: {args.dataset}")
    print(f"Classification par: {args.label}")
    
    if args.mode == "custom":
        min_samples = input("Minimum d'exemples par classe [5]: ").strip() or "5"
        top_classes = input("Nombre de classes [15]: ").strip() or "15"
        
        print("\nVectorisation:")
        print("1. BOW  2. TF-IDF  3. Word2Vec  4. FastText  5. Transformer")
        vectorizer = {"1": "bow", "2": "tfidf", "3": "word2vec", "4": "fasttext", 
                     "5": "transformer"}.get(input("Choix [1]: ").strip() or "1", "bow")
        
        print("\nClassifieur:")
        print("1. Logistique  2. SVM  3. Random Forest")
        classifier = {"1": "logistic", "2": "svm", "3": "random_forest"
                     }.get(input("Choix [1]: ").strip() or "1", "logistic")
        
        use_stopwords = input("Filtrer mots vides? (o/n) [o]: ").strip().lower().startswith("o")
        bpe_merges = input("Fusions BPE [1500]: ").strip() or "1500"
        
        cmd = [
            "python", "main.py",
            "--mode", "all",
            "--label", args.label,
            "--min_samples", min_samples,
            "--top_classes", top_classes,
            "--vectorizers", vectorizer,
            "--classifier", classifier,
            "--bpe_merges", bpe_merges
        ]
        
        if use_stopwords:
            cmd.append("--use_stopwords")
        
    else:
        cmd = [
            "python", "main.py",
            "--mode", "best",
            "--label", args.label
        ]
        print("\nUtilisation des paramètres optimaux")
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
