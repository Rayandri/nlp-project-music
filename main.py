#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classification de paroles de chansons francophones
===============================================

Ce script analyse un corpus de paroles de chansons francophones et
entraîne différents modèles pour prédire l'artiste à partir du texte des paroles.

Processus:
1. Chargement des données (lyrics_dataset)
2. Tokenisation avec BPE
3. Vectorisation avec 5 méthodes différentes
4. Classification et évaluation des performances
"""

import os
import argparse
import numpy as np
from typing import List, Dict

from utils.tokenizer import BPETokenizer
from utils.data_loader import load_lyrics_dataset, save_tokenized_lyrics, get_label_from_metadata
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier, evaluate_multiple_embeddings

def parse_args():
    """
    Parse les arguments de ligne de commande
    
    Returns:
        Namespace avec les arguments
    """
    parser = argparse.ArgumentParser(description="Classification de paroles de chansons")
    
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset",
                        help="Chemin vers le répertoire contenant les paroles")
    
    parser.add_argument("--output_dir", type=str, default="tokenized_lyrics_dataset",
                        help="Chemin pour la sauvegarde des paroles tokenisées")
    
    parser.add_argument("--mode", type=str, choices=["tokenize", "classify", "all"], default="all",
                        help="Mode d'exécution (tokenize, classify ou all)")
    
    parser.add_argument("--label", type=str, choices=["artiste", "album", "genre", "année"], default="artiste",
                        help="Type d'étiquette à prédire")
    
    parser.add_argument("--vectorizers", type=str, nargs="+", 
                        choices=["bow", "tfidf", "word2vec", "fasttext", "transformer", "all"],
                        default=["all"],
                        help="Méthodes de vectorisation à utiliser")
    
    parser.add_argument("--save_vectors", action="store_true",
                        help="Sauvegarder les vecteurs générés au format .npy")
    
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Graine aléatoire pour la reproductibilité")
    
    return parser.parse_args()

def main():
    """
    Fonction principale du script
    """
    args = parse_args()
    
    # 1. Chargement des données
    print("\n=== Chargement des données ===")
    texts, metadata_list = load_lyrics_dataset(args.input_dir)
    print(f"Nombre total de chansons: {len(texts)}")
    
    # Extraire les labels
    labels = get_label_from_metadata(metadata_list, args.label)
    print(f"Nombre de {args.label}s uniques: {len(set(labels))}")
    
    # 2. Tokenisation avec BPE si demandé
    if args.mode in ["tokenize", "all"]:
        print("\n=== Tokenisation des paroles ===")
        
        # Initialiser le tokenizer BPE et l'entraîner sur le corpus
        tokenizer = BPETokenizer(dataset=texts)
        
        # Tokeniser tous les textes
        tokenized_texts = [tokenizer(text) for text in texts]
        print(f"Nombre de textes tokenisés: {len(tokenized_texts)}")
        
        # Sauvegarder les résultats tokenisés
        save_tokenized_lyrics(tokenized_texts, metadata_list, args.output_dir)
    
    # 3 & 4. Vectorisation et classification
    if args.mode in ["classify", "all"]:
        print("\n=== Vectorisation et classification ===")
        
        # Déterminer les méthodes de vectorisation à utiliser
        vectorization_methods = []
        if "all" in args.vectorizers:
            vectorization_methods = ["bow", "tfidf", "word2vec", "fasttext", "transformer"]
        else:
            vectorization_methods = args.vectorizers
        
        # Stocker les résultats pour chaque méthode
        results = {}
        
        for method in vectorization_methods:
            print(f"\nMéthode de vectorisation: {method}")
            
            # Créer et entraîner le vectoriseur
            vectorizer = TextVectorizer(method=method)
            X = vectorizer.fit_transform(texts)
            print(f"Shape des vecteurs: {X.shape}")
            
            # Sauvegarder les vecteurs si demandé
            if args.save_vectors:
                output_dir = "vectors"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{method}_vectors.npy")
                np.save(output_file, X)
                print(f"Vecteurs sauvegardés: {output_file}")
            
            # Entraîner et évaluer le classifieur
            classifier = TextClassifier(model_type="logistic")
            eval_results = classifier.train(
                X, labels, 
                test_size=0.2, 
                random_state=args.random_seed, 
                stratify=True
            )
            
            # Afficher les résultats
            accuracy = eval_results["accuracy"]
            report = eval_results["classification_report"]
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"F1-score macro: {report['macro avg']['f1-score']:.3f}")
            
            # Stocker les résultats pour comparaison
            results[method] = eval_results
        
        # Comparer les performances des différentes méthodes
        if len(results) > 1:
            print("\n=== Comparaison des méthodes de vectorisation ===")
            evaluate_multiple_embeddings(results)

if __name__ == "__main__":
    main() 
