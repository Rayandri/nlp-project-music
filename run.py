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

from utils.tokenizer import BPETokenizer
from utils.data_loader import load_lyrics_dataset, save_tokenized_lyrics, get_label_from_metadata
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier, evaluate_multiple_embeddings, plot_confusion_matrix
from utils.text_generator import TextGenerator, benchmark_generation_models
from utils.data_augmentation import DataAugmenter, augment_dataset, evaluate_augmentation_impact
from utils.model_interpretation import ModelInterpreter

def parse_args():
    parser = argparse.ArgumentParser(description="Outils NLP pour paroles de chansons")
    
    # Mode principal
    parser.add_argument("--mode", type=str, 
                       choices=["classify", "generate", "augment", "interpret", "all"],
                       default="classify", 
                       help="Mode d'opération")
    
    # Chemins des données
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset")
    parser.add_argument("--output_dir", type=str, default="results")
    
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
    
    # Vectoriser et entraîner un modèle
    vectorizer = TextVectorizer(method="tfidf")
    X = vectorizer.fit_transform(texts)
    
    classifier = TextClassifier(model_type=args.classifier)
    classifier.train(X, labels, test_size=0.2, random_state=args.random_seed)
    
    # Créer l'interpréteur
    interpreter = ModelInterpreter(classifier.model, vectorizer)
    
    # Interprétation basée sur les coefficients
    if "coefficients" in args.interpretation and hasattr(classifier.model, "coef_"):
        print("\nAnalyse de l'importance des features (coefficients):")
        top_features = interpreter.get_feature_importance(method="coefficients")
        for feature, importance in top_features[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        # Générer un nuage de mots
        plt.figure(figsize=(10, 6))
        wordcloud_fig = interpreter.plot_word_cloud(top_n=200)
        wordcloud_path = os.path.join(args.output_dir, "feature_importance_wordcloud.png")
        wordcloud_fig.savefig(wordcloud_path)
        print(f"Nuage de mots sauvegardé: {wordcloud_path}")
    
    # Interprétation basée sur la permutation
    if "permutation" in args.interpretation:
        print("\nAnalyse de l'importance des features (permutation):")
        try:
            perm_importance = interpreter.permutation_feature_importance(X, labels, n_repeats=5)
            for feature, importance in perm_importance[:10]:
                print(f"  {feature}: {importance:.4f}")
        except Exception as e:
            print(f"Erreur lors du calcul de l'importance par permutation: {e}")
    
    # Exemple d'explication d'une prédiction
    if texts:
        example_text = texts[0][:500]  # Limiter la longueur pour l'affichage
        print(f"\nExplication de prédiction pour un exemple:")
        try:
            explanation = interpreter.explain_prediction(example_text, num_features=5)
            print(f"  Prédiction: {explanation['prediction']}")
            print("  Top features contribuant à la prédiction:")
            for feature, contribution in explanation["top_features"]:
                print(f"    {feature}: {contribution:.4f}")
        except Exception as e:
            print(f"Erreur lors de l'explication de la prédiction: {e}")

if __name__ == "__main__":
    main() 
