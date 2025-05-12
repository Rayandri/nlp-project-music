#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from utils.tokenizer import BPETokenizer
from utils.data_loader import load_lyrics_dataset, save_tokenized_lyrics, get_label_from_metadata
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier, evaluate_multiple_embeddings, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Lyrics classification")
    
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset")
    parser.add_argument("--output_dir", type=str, default="tokenized_lyrics_dataset")
    
    # Mode de fonctionnement
    parser.add_argument("--mode", type=str, 
                        choices=["tokenize", "classify", "all", "best"],
                        default="best", 
                        help="Mode: 'best' utilise les paramètres optimaux")
    
    # Paramètres de base
    parser.add_argument("--label", type=str, 
                        choices=["artiste", "album", "genre", "année"], 
                        default="artiste")
    parser.add_argument("--vectorizers", type=str, nargs="+", 
                        choices=["bow", "tfidf", "word2vec", "fasttext", "transformer", "all"],
                        default=["bow"])
    parser.add_argument("--classifier", type=str, 
                        choices=["logistic", "svm", "random_forest"],
                        default="logistic")
    
    # Paramètres avancés (avec valeurs optimales par défaut)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_samples", type=int, default=5, 
                        help="Minimum d'exemples par classe")
    parser.add_argument("--top_classes", type=int, default=15, 
                        help="Nombre de classes les plus fréquentes à conserver")
    parser.add_argument("--bpe_merges", type=int, default=1000, 
                        help="Nombre de fusions BPE")
    parser.add_argument("--use_stopwords", action="store_true", 
                        help="Filtrer les mots vides")
    parser.add_argument("--pca", type=int, default=0, 
                        help="Réduction de dimension PCA (0=désactivé)")
    
    # Options supplémentaires
    parser.add_argument("--save_vectors", action="store_true", 
                        help="Sauvegarder les vecteurs")
    parser.add_argument("--confusion_matrix", action="store_true", default=True, 
                        help="Générer la matrice de confusion")
    parser.add_argument("--save_models", type=str, 
                        help="Sauvegarder les modèles entraînés dans le répertoire spécifié")
    
    args = parser.parse_args()
    
    # Mode "best": paramètres optimaux
    if args.mode == "best":
        args.mode = "all"
        args.min_samples = 5
        args.top_classes = 15
        args.classifier = "logistic"
        args.vectorizers = ["bow"]
        args.use_stopwords = True
        args.bpe_merges = 1500
        args.confusion_matrix = True
        print("Mode 'best': utilisation des paramètres optimaux")
        
    return args

def save_models(tokenizer, vectorizer, classifier, labels, output_dir):
    """Sauvegarde les modèles entraînés pour une utilisation ultérieure"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer un mapping des labels pour la prédiction
    unique_labels = sorted(list(set(labels)))
    label_map = {i: label for i, label in enumerate(unique_labels)}
    
    # Sauvegarder les modèles
    with open(os.path.join(output_dir, "tokenizer.pkl"), 'wb') as f:
        pickle.dump(tokenizer, f)
        
    with open(os.path.join(output_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(output_dir, "classifier.pkl"), 'wb') as f:
        pickle.dump(classifier, f)
        
    with open(os.path.join(output_dir, "labels.pkl"), 'wb') as f:
        pickle.dump(label_map, f)
        
    print(f"Modèles sauvegardés dans {output_dir}")
    print(f"Nombre de classes: {len(unique_labels)}")

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    
    print("\n=== Chargement des données ===")
    texts, metadata_list = load_lyrics_dataset(args.input_dir)
    print(f"Total chansons: {len(texts)}")
    
    labels = get_label_from_metadata(metadata_list, args.label)
    print(f"{args.label}s uniques: {len(set(labels))}")
    
    # Filtrer les classes avec trop peu d'échantillons
    if args.min_samples > 1:
        from collections import Counter
        label_counts = Counter(labels)
        valid_labels = {label for label, count in label_counts.items() if count >= args.min_samples}
        
        filtered_indices = [i for i, label in enumerate(labels) if label in valid_labels]
        texts = [texts[i] for i in filtered_indices]
        metadata_list = [metadata_list[i] for i in filtered_indices]
        labels = [labels[i] for i in filtered_indices]
        
        print(f"Filtré à {len(texts)} chansons avec au moins {args.min_samples} exemples par {args.label}")
        print(f"{args.label}s uniques restants: {len(valid_labels)}")
    
    # Conserver uniquement les N classes les plus fréquentes
    if args.top_classes > 0:
        from collections import Counter
        label_counts = Counter(labels)
        top_labels = {label for label, _ in label_counts.most_common(args.top_classes)}
        
        filtered_indices = [i for i, label in enumerate(labels) if label in top_labels]
        texts = [texts[i] for i in filtered_indices]
        metadata_list = [metadata_list[i] for i in filtered_indices]
        labels = [labels[i] for i in filtered_indices]
        
        print(f"Filtré à {len(texts)} chansons des {args.top_classes} {args.label}s les plus fréquents")
        print(f"{args.label}s uniques restants: {len(set(labels))}")
    
    # Initialiser le tokenizer
    if args.mode in ["tokenize", "all"]:
        print("\n=== Tokenisation des paroles ===")
        tokenizer = BPETokenizer(
            dataset=texts, 
            num_merges=args.bpe_merges,
            use_stopwords=args.use_stopwords
        )
        tokenized_texts = [tokenizer(text) for text in texts]
        print(f"Textes tokenisés: {len(tokenized_texts)}")
        save_tokenized_lyrics(tokenized_texts, metadata_list, args.output_dir)
    else:
        # Si on ne tokenize pas, il faut quand même créer le tokenizer
        tokenizer = BPETokenizer(
            num_merges=args.bpe_merges,
            use_stopwords=args.use_stopwords
        )
    
    # Variables pour stocker le meilleur modèle
    best_vectorizer = None
    best_classifier = None
    
    if args.mode in ["classify", "all"]:
        print("\n=== Vectorisation et classification ===")
        vectorization_methods = []
        if "all" in args.vectorizers:
            vectorization_methods = ["bow", "tfidf", "word2vec", "fasttext", "transformer"]
        else:
            vectorization_methods = args.vectorizers
        
        results = {}
        best_accuracy = 0
        best_method = None
        best_X = None
        
        for method in vectorization_methods:
            print(f"\nMéthode de vectorisation: {method}")
            vectorizer = TextVectorizer(method=method)
            X = vectorizer.fit_transform(texts)
            print(f"Dimensions des vecteurs: {X.shape}")
            
            # Réduction de dimensionnalité optionnelle
            if args.pca > 0 and X.shape[1] > args.pca:
                pca = PCA(n_components=args.pca)
                X_reduced = pca.fit_transform(X)
                explained_var = sum(pca.explained_variance_ratio_) * 100
                print(f"PCA appliquée: {X.shape[1]} -> {args.pca} dimensions ({explained_var:.2f}% variance conservée)")
                X = X_reduced
                
                # Mettre à jour le vectorizer pour la prédiction
                original_transform = vectorizer.transform
                vectorizer.transform = lambda texts: pca.transform(original_transform(texts))
            
            if args.save_vectors:
                output_dir = "vectors"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{method}_vectors.npy")
                np.save(output_file, X)
                print(f"Vecteurs sauvegardés: {output_file}")
            
            classifier = TextClassifier(model_type=args.classifier)
            eval_results = classifier.train(
                X, labels, 
                test_size=0.2, 
                random_state=args.random_seed, 
                stratify=True
            )
            
            accuracy = eval_results["accuracy"]
            report = eval_results["classification_report"]
            
            print(f"Précision: {accuracy:.3f}")
            print(f"F1-score macro: {report['macro avg']['f1-score']:.3f}")
            
            results[method] = eval_results
            
            # Tracker la meilleure méthode
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
                best_X = X
                best_vectorizer = vectorizer
                best_classifier = classifier
        
        if len(results) > 1:
            print("\n=== Comparaison des méthodes de vectorisation ===")
            evaluate_multiple_embeddings(results)
            print(f"\nMeilleure méthode: {best_method} (Précision: {best_accuracy:.3f})")
        
        # Générer la matrice de confusion pour le meilleur modèle
        if args.confusion_matrix and best_method:
            best_results = results[best_method]
            cm = best_results["confusion_matrix"]
            
            # Obtenir les classes qui apparaissent dans le jeu de test
            y_test = best_results["y_test"]
            y_pred = best_results["y_pred"]
            classes = sorted(list(set(y_test)))
            
            # Si trop de classes, limiter aux 15 plus fréquentes
            if len(classes) > 15:
                from collections import Counter
                class_counts = Counter(y_test)
                classes = [c for c, _ in class_counts.most_common(15)]
                
                # Filtrer la matrice de confusion pour inclure uniquement ces classes
                indices = [i for i, c in enumerate(y_test) if c in classes]
                y_test_filtered = [y_test[i] for i in indices]
                y_pred_filtered = [y_pred[i] for i in indices]
                cm = confusion_matrix(y_test_filtered, y_pred_filtered)
            
            # Créer le répertoire de sortie pour les résultats
            results_dir = "results_rapport"
            os.makedirs(results_dir, exist_ok=True)
            
            # Sauvegarder la matrice de confusion
            plot_confusion_matrix(cm, classes, title=f"Matrice de confusion - {best_method}", 
                                 output_dir=results_dir)
            
            # Créer un résumé des résultats
            summary_file = os.path.join(results_dir, "main_classification_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=== RÉSULTATS DE CLASSIFICATION (main.py) ===\n\n")
                f.write(f"Meilleure méthode: {best_method}\n")
                f.write(f"Précision: {best_accuracy:.4f}\n")
                f.write(f"F1-score macro: {results[best_method]['classification_report']['macro avg']['f1-score']:.4f}\n")
                f.write(f"F1-score pondéré: {results[best_method]['classification_report']['weighted avg']['f1-score']:.4f}\n")
                f.write(f"Nombre de classes: {len(set(labels))}\n")
                f.write(f"Nombre d'échantillons: {len(texts)}\n\n")
                
                # Ajouter les détails de chaque méthode
                f.write("=== Détails par méthode ===\n")
                for method, res in results.items():
                    f.write(f"\nMéthode: {method}\n")
                    f.write(f"Précision: {res['accuracy']:.4f}\n")
                    f.write(f"F1-score macro: {res['classification_report']['macro avg']['f1-score']:.4f}\n")
                
            print(f"Résumé des résultats sauvegardé: {summary_file}")
            
            # Afficher les résultats finaux
            print("\n=== RÉSULTATS FINAUX ===")
            print(f"Meilleure méthode: {best_method}")
            print(f"Précision: {best_accuracy:.3f}")
            print(f"F1-score macro: {results[best_method]['classification_report']['macro avg']['f1-score']:.3f}")
            print(f"Nombre de classes: {len(set(labels))}")
            print(f"Nombre d'échantillons: {len(texts)}")
            print("--------------------------------------------------")
        
        # Évaluer toutes les méthodes sur un graphique de comparaison
        if len(results) > 1:
            # Créer le répertoire si nécessaire
            results_dir = "results_rapport"
            os.makedirs(results_dir, exist_ok=True)
            
            # Évaluer avec la fonction améliorée
            evaluate_multiple_embeddings(results, output_dir=results_dir)
        
        # Sauvegarder les modèles si demandé
        if args.save_models and best_vectorizer and best_classifier:
            save_models(tokenizer, best_vectorizer, best_classifier, labels, args.save_models)

if __name__ == "__main__":
    main() 
