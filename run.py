#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour lancer différentes tâches du projet
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time  # Add time module for logging

from utils.tokenizer import BPETokenizer
from utils.data_loader import load_lyrics_dataset, save_tokenized_lyrics, get_label_from_metadata, cross_dataset_validation
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier, evaluate_multiple_embeddings, plot_confusion_matrix
from utils.text_generator import TextGenerator, benchmark_generation_models
from utils.data_augmentation import DataAugmenter, augment_dataset, evaluate_augmentation_impact
from utils.model_interpretation import ModelInterpreter

def parse_args():
    parser = argparse.ArgumentParser(description="Outils NLP pour paroles de chansons")
    
    # Mode principal
    parser.add_argument("--mode", type=str, 
                       choices=["classify", "generate", "augment", "interpret", "cross_validate", "all"],
                       default="classify", 
                       help="Mode d'opération")
    
    # Chemins des données
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--dataset_dirs", type=str, nargs="+", 
                       help="Répertoires des jeux de données pour la validation croisée")
    
    # Paramètres généraux
    parser.add_argument("--label", type=str, 
                       choices=["artiste", "album", "genre", "année"], 
                       default="artiste")
    parser.add_argument("--random_seed", type=int, default=42)
    
    # Paramètres de classification
    parser.add_argument("--classifier", type=str, 
                       choices=["logistic", "svm", "random_forest"],
                       default="logistic")
    parser.add_argument("--vectorizers", type=str, nargs="+", 
                       choices=["bow", "tfidf", "word2vec", "fasttext", "transformer", "all"],
                       default=["tfidf"])
    
    # Paramètres de génération
    parser.add_argument("--generator", type=str, 
                       choices=["ngram", "word2vec", "fasttext", "transformer", "all"],
                       default="ngram")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Longueur maximale du texte généré")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Température pour la génération (plus élevée = plus aléatoire)")
    
    # Paramètres d'augmentation
    parser.add_argument("--augmentation", type=str, nargs="+",
                       choices=["random_deletion", "random_swap", "random_insertion",
                                "synonym_replacement", "back_translation", "contextual_augmentation", "all"],
                       default=["all"])
    parser.add_argument("--augmentation_factor", type=float, default=0.5,
                       help="Facteur d'augmentation (par rapport à la taille du dataset original)")
    
    # Paramètres d'interprétation
    parser.add_argument("--interpretation", type=str, nargs="+",
                       choices=["coefficients", "permutation", "lime", "shap", "all"],
                       default=["coefficients"])
    
    args = parser.parse_args()
    
    # Traiter les options "all"
    if "all" in args.vectorizers:
        args.vectorizers = ["bow", "tfidf", "word2vec", "fasttext", "transformer"]
        
    if "all" in args.augmentation:
        args.augmentation = ["random_deletion", "random_swap", "random_insertion",
                            "synonym_replacement", "back_translation", "contextual_augmentation"]
        
    if "all" in args.interpretation:
        args.interpretation = ["coefficients", "permutation", "lime", "shap"]
        
    if args.generator == "all":
        args.generator = ["ngram", "word2vec", "fasttext", "transformer"]
    else:
        args.generator = [args.generator]
        
    return args

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Si on n'est pas en mode cross-validate, on charge les données normalement
    if args.mode != "cross_validate":
        print("\n=== Chargement des données ===")
        texts, metadata_list = load_lyrics_dataset(args.input_dir)
        print(f"Total chansons: {len(texts)}")
        
        # Récupérer les labels
        labels = get_label_from_metadata(metadata_list, args.label)
        print(f"{args.label}s uniques: {len(set(labels))}")
        
        # Afficher la distribution des labels
        label_counter = Counter(labels)
        print(f"Top 5 {args.label}s: {label_counter.most_common(5)}")
        
        # Exécuter le mode choisi
        if args.mode in ["classify", "all"]:
            run_classification(texts, labels, args)
            
        if args.mode in ["generate", "all"]:
            run_generation(texts, args)
            
        if args.mode in ["augment", "all"]:
            run_augmentation(texts, labels, args)
            
        if args.mode in ["interpret", "all"]:
            run_interpretation(texts, labels, args)
    else:
        # Mode validation croisée
        run_cross_validation(args)

def run_classification(texts, labels, args):
    print("\n=== Mode Classification ===")
    
    results = {}
    best_accuracy = 0
    best_method = None
    
    for method in args.vectorizers:
        print(f"\nMéthode de vectorisation: {method}")
        vectorizer = TextVectorizer(method=method)
        X = vectorizer.fit_transform(texts)
        print(f"Dimensions des vecteurs: {X.shape}")
        
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
    
    if len(results) > 1:
        print("\n=== Comparaison des méthodes de vectorisation ===")
        evaluate_multiple_embeddings(results)
        print(f"\nMeilleure méthode: {best_method} (Précision: {best_accuracy:.3f})")
        
    # Sauvegarder le résultat pour la visualisation
    output_file = os.path.join(args.output_dir, "classification_results.npy")
    np.save(output_file, results)
    print(f"Résultats sauvegardés: {output_file}")

def run_generation(texts, args):
    print("\n=== Mode Génération ===")
    
    # Effectuer un benchmark des modèles de génération
    results, samples = benchmark_generation_models(texts, generator_types=args.generator)
    
    # Afficher quelques exemples générés
    for generator_type, texts_list in samples.items():
        print(f"\nExemples générés avec {generator_type}:")
        for i, text in enumerate(texts_list[:2]):  # Limiter à 2 exemples
            print(f"  Exemple {i+1}: {text[:100]}...")
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "generation_results.npy")
    np.save(output_file, {"results": results, "samples": samples})
    print(f"Résultats sauvegardés: {output_file}")

def run_augmentation(texts, labels, args):
    print("\n=== Mode Augmentation ===")
    
    # Évaluer l'impact de l'augmentation de données
    augmentation_results = evaluate_augmentation_impact(
        texts, labels, 
        augmentation_methods=args.augmentation,
        classifier_type=args.classifier
    )
    
    # Afficher les résultats
    baseline = augmentation_results["baseline"]
    print(f"Baseline (sans augmentation):")
    print(f"  Précision: {baseline['accuracy']:.3f}")
    print(f"  F1-score: {baseline['f1_score']:.3f}")
    print(f"  Taille du dataset: {baseline['train_size']}")
    
    for factor in sorted([k for k in augmentation_results.keys() if k != "baseline"]):
        result = augmentation_results[factor]
        print(f"\nAvec augmentation (facteur {factor.split('_')[1]}):")
        print(f"  Précision: {result['accuracy']:.3f}")
        print(f"  F1-score: {result['f1_score']:.3f}")
        print(f"  Taille du dataset: {result['train_size']}")
        print(f"  Amélioration: {result['improvement']*100:.2f}%")
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "augmentation_results.npy")
    np.save(output_file, augmentation_results)
    print(f"Résultats sauvegardés: {output_file}")

def run_interpretation(texts, labels, args):
    print("\n=== Mode Interprétation ===")
    
    start_total = time.time()
    
    # Vectoriser et entraîner un modèle
    print("Vectorizing data...")
    start_vectorize = time.time()
    vectorizer = TextVectorizer(method="tfidf")
    X = vectorizer.fit_transform(texts)
    print(f"Vectorization completed in {time.time() - start_vectorize:.2f} seconds")
    
    print("Training classifier...")
    start_train = time.time()
    classifier = TextClassifier(model_type=args.classifier)
    classifier.train(X, labels, test_size=0.2, random_state=args.random_seed)
    print(f"Training completed in {time.time() - start_train:.2f} seconds")
    
    # Créer l'interpréteur
    interpreter = ModelInterpreter(classifier.model, vectorizer)
    
    # Interprétation basée sur les coefficients
    if "coefficients" in args.interpretation and hasattr(classifier.model, "coef_"):
        print("\nAnalyse de l'importance des features (coefficients):")
        start_coef = time.time()
        try:
            top_features = interpreter.get_feature_importance(method="coefficients")
            if top_features:
                for feature, importance in top_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
                
                # Générer un nuage de mots si on a des features
                try:
                    plt.figure(figsize=(10, 6))
                    wordcloud_fig = interpreter.plot_word_cloud(top_n=200)
                    wordcloud_path = os.path.join(args.output_dir, "feature_importance_wordcloud.png")
                    wordcloud_fig.savefig(wordcloud_path)
                    print(f"Nuage de mots sauvegardé: {wordcloud_path}")
                except Exception as e:
                    print(f"Erreur lors de la génération du nuage de mots: {str(e)}")
        except Exception as e:
            print(f"Erreur lors de l'analyse de l'importance des features: {str(e)}")
        print(f"Coefficient importance analysis completed in {time.time() - start_coef:.2f} seconds")
    
    # Interprétation basée sur la permutation
    if "permutation" in args.interpretation:
        print("\nAnalyse de l'importance des features (permutation):")
        start_perm = time.time()
        try:
            # Utiliser un sous-ensemble plus petit pour accélérer tout en gardant assez d'exemples
            sample_size = min(500, int(X.shape[0] * 0.25))
            
            # Assurer une représentation équilibrée des classes si possible
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                
                # Utiliser un échantillonnage stratifié pour préserver la distribution des classes
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=args.random_seed)
                for train_idx, _ in splitter.split(X, labels):
                    indices = train_idx
                    break
            except:
                # En cas d'échec, utiliser l'échantillonnage simple
                indices = np.random.choice(X.shape[0], sample_size, replace=False)
                
            X_sample = X[indices]
            y_sample = [labels[i] for i in indices]
            
            print(f"Running permutation importance on {sample_size} samples (classes: {len(set(y_sample))})...")
            perm_features = interpreter.permutation_feature_importance(X_sample, y_sample, n_repeats=5)
            
            if perm_features:
                for feature, importance in perm_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
                
                # Comparer avec les résultats des coefficients
                if "coefficients" in args.interpretation and hasattr(classifier.model, "coef_"):
                    print("\nComparaison des méthodes d'importance:")
                    coef_features = set([f for f, _ in top_features[:20]])
                    perm_features_set = set([f for f, _ in perm_features[:20]])
                    overlap = coef_features.intersection(perm_features_set)
                    print(f"Chevauchement dans le top 20: {len(overlap)} features")
        except Exception as e:
            print(f"Erreur lors de l'analyse de l'importance par permutation: {str(e)}")
        print(f"Permutation importance analysis completed in {time.time() - start_perm:.2f} seconds")
    
    # Interprétation avec LIME
    if "lime" in args.interpretation:
        print("\nExplication avec LIME:")
        start_lime = time.time()
        try:
            # Choisir un exemple aléatoire
            sample_idx = np.random.randint(0, len(texts))
            sample_text = texts[sample_idx]
            sample_label = labels[sample_idx]
            
            # Obtenir l'explication LIME
            lime_explanation = interpreter.explain_with_lime(sample_text, num_features=10)
            
            print(f"Texte: {sample_text[:100]}...")
            print(f"Label réel: {sample_label}")
            print(f"Prédiction: {lime_explanation['prediction']}")
            print("Top features explicatives:")
            for feature, weight in lime_explanation["explanations"]:
                print(f"  {feature}: {weight:.4f}")
        except Exception as e:
            print(f"Erreur lors de l'explication avec LIME: {str(e)}")
        print(f"LIME explanation completed in {time.time() - start_lime:.2f} seconds")
    
    # Interprétation avec SHAP
    if "shap" in args.interpretation:
        print("\nExplication avec SHAP:")
        start_shap = time.time()
        try:
            # Utiliser un sous-ensemble pour accélérer
            sample_size = min(100, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            # Calculer les valeurs SHAP
            shap_values = interpreter.explain_with_shap(X_sample, num_samples=50)
            
            # Afficher les résultats
            print("Analyse SHAP complétée.")
        except Exception as e:
            print(f"Erreur lors de l'explication avec SHAP: {str(e)}")
        print(f"SHAP explanation completed in {time.time() - start_shap:.2f} seconds")
    
    print(f"\nTotal interpretation time: {time.time() - start_total:.2f} seconds")

def run_cross_validation(args):
    """Mode de validation croisée entre datasets"""
    print("\n=== Mode Validation Croisée ===")
    
    # Vérifier qu'on a bien les répertoires de datasets
    if not args.dataset_dirs or len(args.dataset_dirs) < 2:
        print("Erreur: Le mode cross_validate nécessite au moins 2 datasets.")
        print("Exemple: --dataset_dirs dataset1 dataset2")
        return
        
    # Exécuter la validation croisée
    results = cross_dataset_validation(
        dataset_dirs=args.dataset_dirs,
        vectorizer_type=args.vectorizers[0] if args.vectorizers else "tfidf",
        classifier_type=args.classifier,
        label_type=args.label
    )
    
    # Afficher les résultats sous forme de tableau
    print("\nRésultats de la validation croisée:")
    print("-" * 80)
    print(f"{'Train → Test':<20} {'Précision':<10} {'F1 Macro':<10} {'Classes':<10} {'Train':<8} {'Test':<8}")
    print("-" * 80)
    
    for key, result in results.items():
        if "error" in result:
            print(f"{key:<20} {result['error']}")
            continue
            
        train, test = key.split("_to_")
        train_base = os.path.basename(train)
        test_base = os.path.basename(test)
        display_key = f"{train_base} → {test_base}"
        
        accuracy = result.get("accuracy", 0)
        f1 = result.get("macro_f1", 0)
        common_classes = result.get("common_classes", 0)
        train_size = result.get("train_size", 0)
        test_size = result.get("test_size", 0)
        
        print(f"{display_key:<20} {accuracy:.3f}{'':<5} {f1:.3f}{'':<5} {common_classes:<10} {train_size:<8} {test_size:<8}")
    
    print("-" * 80)
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "cross_validation_results.npy")
    np.save(output_file, results)
    print(f"Résultats sauvegardés: {output_file}")

if __name__ == "__main__":
    main() 
