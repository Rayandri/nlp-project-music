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
    best_vectorizer = None
    best_classifier = None
    
    for method in args.vectorizers:
        print(f"\nMéthode de vectorisation: {method}")
        
        # Options de vectorisation pour limiter l'utilisation de la mémoire
        vectorizer_opts = {}
        if method in ["tfidf", "bow"]:
            # Limiter le nombre de features et réduire la mémoire utilisée
            vectorizer_opts = {
                "max_features": 10000,  # Limiter à 10000 features au lieu de toutes
                "min_df": 2,            # Ignorer les mots qui apparaissent dans moins de 2 documents
                "max_df": 0.9           # Ignorer les mots qui apparaissent dans plus de 90% des documents
            }
        
        vectorizer = TextVectorizer(method=method, **vectorizer_opts)
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
        
        try:
            # Sauvegarder le modèle et le vectorizer
            print(f"\nSauvegarde du modèle {method}_{args.classifier}_{args.label}...")
            models_dir = os.path.join(args.output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            print(f"Répertoire des modèles: {models_dir}")
            model_path = os.path.join(models_dir, f"{method}_{args.classifier}_{args.label}.pkl")
            print(f"Chemin complet du modèle: {model_path}")
            print(f"Le modèle est-il valide: {classifier.model is not None}")
            classifier.save_model(model_path, vectorizer)
            print(f"Modèle et vectorizeur sauvegardés dans: {model_path}")
        except Exception as e:
            print(f"ERREUR lors de la sauvegarde du modèle: {str(e)}")
        
        # Tracker la meilleure méthode
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method
            best_vectorizer = vectorizer
            best_classifier = classifier
            
        # Sauvegarder la matrice de confusion pour chaque méthode
        y_test = eval_results["y_test"]
        y_pred = eval_results["y_pred"]
        cm = eval_results["confusion_matrix"]
        
        # Obtenir les classes uniques (limiter à 15 si trop nombreuses)
        classes = sorted(list(set(y_test)))
        if len(classes) > 15:
            from collections import Counter
            class_counts = Counter(y_test)
            classes = [c for c, _ in class_counts.most_common(15)]
        
        # Sauvegarder la matrice de confusion
        title = f"Confusion Matrix - {method}"
        plot_confusion_matrix(cm, classes, title=title, output_dir=args.output_dir)
    
    # Sauvegarder le meilleur modèle séparément
    if best_classifier and best_vectorizer:
        best_model_path = os.path.join(models_dir, f"best_{args.label}.pkl")
        best_classifier.save_model(best_model_path, best_vectorizer)
        print(f"Meilleur modèle sauvegardé dans: {best_model_path}")
    
    if len(results) > 1:
        print("\n=== Comparaison des méthodes de vectorisation ===")
        evaluate_multiple_embeddings(results, output_dir=args.output_dir)
        print(f"\nMeilleure méthode: {best_method} (Précision: {best_accuracy:.3f})")
        
    # Sauvegarder les résultats détaillés pour le rapport
    output_file = os.path.join(args.output_dir, "classification_results.npy")
    np.save(output_file, results)
    
    # Sauvegarder aussi un résumé des résultats en format texte pour faciliter l'inclusion dans le rapport
    summary_file = os.path.join(args.output_dir, "classification_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Résultats de Classification ===\n\n")
        for method, res in results.items():
            f.write(f"Méthode: {method}\n")
            f.write(f"Précision: {res['accuracy']:.4f}\n")
            f.write(f"F1-score macro: {res['classification_report']['macro avg']['f1-score']:.4f}\n")
            f.write(f"F1-score pondéré: {res['classification_report']['weighted avg']['f1-score']:.4f}\n")
            f.write(f"Cross-validation: {res['cv_scores'].mean():.4f} ± {res['cv_scores'].std():.4f}\n\n")
        f.write(f"Meilleure méthode: {best_method} (Précision: {best_accuracy:.4f})\n")
    
    print(f"Résultats sauvegardés: {output_file}")
    print(f"Résumé des résultats sauvegardé: {summary_file}")

def run_generation(texts, args):
    print("\n=== Mode Génération ===")
    
    # Effectuer un benchmark des modèles de génération
    results, samples = benchmark_generation_models(texts, generator_types=args.generator)
    
    # Résumé des résultats pour le rapport
    summary_file = os.path.join(args.output_dir, "generation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Résultats de Génération de Texte ===\n\n")
        
        # Écrire les résultats de perplexité
        f.write("Perplexité des modèles:\n")
        for generator_type, perplexity in results.items():
            f.write(f"  {generator_type}: {perplexity:.2f}\n")
        f.write("\n")
    
    # Afficher quelques exemples générés
    for generator_type, texts_list in samples.items():
        print(f"\nExemples générés avec {generator_type}:")
        
        # Sauvegarder des exemples plus complets pour le rapport
        examples_file = os.path.join(args.output_dir, f"generated_samples_{generator_type}.txt")
        with open(examples_file, 'w') as f:
            f.write(f"Exemples générés par le modèle {generator_type}:\n\n")
            for i, text in enumerate(texts_list[:5]):  # Enregistrer 5 exemples complets
                f.write(f"Exemple {i+1}:\n{text}\n\n" + "-"*50 + "\n\n")
        
        # Ajouter quelques exemples au résumé
        with open(summary_file, 'a') as f:
            f.write(f"Exemples générés avec {generator_type}:\n")
            for i, text in enumerate(texts_list[:3]):  # Limiter à 3 exemples pour le résumé
                f.write(f"  Exemple {i+1}: {text[:150]}...\n")
            f.write("\n")
        
        # Afficher des extraits dans la console
        for i, text in enumerate(texts_list[:2]):  # Limiter à 2 exemples pour l'affichage
            print(f"  Exemple {i+1}: {text[:100]}...")
        
        # Enregistrer que les exemples ont été sauvegardés
        print(f"  Exemples complets sauvegardés dans: {examples_file}")
    
    # Créer un graphique comparatif de perplexité (si disponible)
    try:
        plt.figure(figsize=(10, 6))
        
        # Collecter les valeurs de perplexité valides
        valid_generators = []
        valid_perplexities = []
        
        for generator_type, perplexity in results.items():
            if perplexity > 0 and perplexity < float('inf'):
                valid_generators.append(generator_type)
                valid_perplexities.append(perplexity)
        
        if valid_generators:
            # Créer le graphique en barres
            plt.bar(valid_generators, valid_perplexities, color='salmon')
            plt.xlabel('Modèle de génération')
            plt.ylabel('Perplexité (échelle log)')
            plt.title('Comparaison de la perplexité des modèles de génération')
            plt.yscale('log')  # Échelle logarithmique pour mieux visualiser
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Sauvegarder le graphique
            perplexity_path = os.path.join(args.output_dir, "generation_perplexity.png")
            plt.savefig(perplexity_path, dpi=300, bbox_inches='tight')
            print(f"\nGraphique de perplexité sauvegardé: {perplexity_path}")
            plt.show()
    except Exception as e:
        print(f"Erreur lors de la génération du graphique de perplexité: {str(e)}")
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "generation_results.npy")
    np.save(output_file, {"results": results, "samples": samples})
    print(f"Résultats sauvegardés: {output_file}")
    print(f"Résumé des résultats sauvegardé: {summary_file}")

def run_augmentation(texts, labels, args):
    print("\n=== Mode Augmentation ===")
    
    # Évaluer l'impact de l'augmentation de données
    augmentation_results = evaluate_augmentation_impact(
        texts, labels, 
        augmentation_methods=args.augmentation,
        classifier_type=args.classifier
    )
    
    # Résumé des résultats pour le rapport
    summary_file = os.path.join(args.output_dir, "augmentation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Résultats d'Augmentation de Données ===\n\n")
    
    # Afficher les résultats
    baseline = augmentation_results["baseline"]
    print(f"Baseline (sans augmentation):")
    print(f"  Précision: {baseline['accuracy']:.3f}")
    print(f"  F1-score: {baseline['f1_score']:.3f}")
    print(f"  Taille du dataset: {baseline['train_size']}")
    
    # Enregistrer les résultats de la baseline
    with open(summary_file, 'a') as f:
        f.write("Baseline (sans augmentation):\n")
        f.write(f"  Précision: {baseline['accuracy']:.4f}\n")
        f.write(f"  F1-score: {baseline['f1_score']:.4f}\n")
        f.write(f"  Taille du dataset: {baseline['train_size']}\n\n")
    
    # Préparer les données pour le graphique
    factors = []
    accuracies = [baseline['accuracy']]
    f1_scores = [baseline['f1_score']]
    dataset_sizes = [baseline['train_size']]
    labels_graph = ['Baseline']
    
    for factor in sorted([k for k in augmentation_results.keys() if k != "baseline"]):
        result = augmentation_results[factor]
        factor_value = float(factor.split('_')[1])
        factors.append(factor_value)
        
        print(f"\nAvec augmentation (facteur {factor.split('_')[1]}):")
        print(f"  Précision: {result['accuracy']:.3f}")
        print(f"  F1-score: {result['f1_score']:.3f}")
        print(f"  Taille du dataset: {result['train_size']}")
        print(f"  Amélioration: {result['improvement']*100:.2f}%")
        
        # Enregistrer les résultats pour chaque facteur
        with open(summary_file, 'a') as f:
            f.write(f"Avec augmentation (facteur {factor.split('_')[1]}):\n")
            f.write(f"  Précision: {result['accuracy']:.4f}\n")
            f.write(f"  F1-score: {result['f1_score']:.4f}\n")
            f.write(f"  Taille du dataset: {result['train_size']}\n")
            f.write(f"  Amélioration: {result['improvement']*100:.2f}%\n\n")
        
        # Collecter les données pour le graphique
        accuracies.append(result['accuracy'])
        f1_scores.append(result['f1_score'])
        dataset_sizes.append(result['train_size'])
        labels_graph.append(f"Aug. {factor_value}")
    
    # Créer un graphique d'impact de l'augmentation sur la précision
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(labels_graph, accuracies, color='skyblue')
    plt.ylabel('Précision')
    plt.title('Impact sur la précision')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(labels_graph, f1_scores, color='lightgreen')
    plt.ylabel('F1-score')
    plt.title('Impact sur le F1-score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    aug_impact_path = os.path.join(args.output_dir, "augmentation_impact.png")
    plt.savefig(aug_impact_path, dpi=300, bbox_inches='tight')
    print(f"Graphique d'impact de l'augmentation sauvegardé: {aug_impact_path}")
    plt.show()
    
    # Graphique de taille du dataset vs. performance
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, accuracies, 'o-', color='blue', label='Précision')
    plt.plot(dataset_sizes, f1_scores, 's-', color='green', label='F1-score')
    
    plt.xlabel('Taille du dataset')
    plt.ylabel('Performance')
    plt.title('Performance vs. Taille du dataset')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    dataset_size_path = os.path.join(args.output_dir, "dataset_size_vs_performance.png")
    plt.savefig(dataset_size_path, dpi=300, bbox_inches='tight')
    print(f"Graphique taille vs. performance sauvegardé: {dataset_size_path}")
    plt.show()
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "augmentation_results.npy")
    np.save(output_file, augmentation_results)
    print(f"Résultats sauvegardés: {output_file}")
    print(f"Résumé des résultats sauvegardé: {summary_file}")

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
    
    # Fichier de résumé pour le rapport
    summary_file = os.path.join(args.output_dir, "interpretation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Résultats d'Interprétation du Modèle ===\n\n")
    
    # Interprétation basée sur les coefficients
    if "coefficients" in args.interpretation and hasattr(classifier.model, "coef_"):
        print("\nAnalyse de l'importance des features (coefficients):")
        start_coef = time.time()
        try:
            top_features = interpreter.get_feature_importance(method="coefficients")
            if top_features:
                # Enregistrer dans le fichier de résumé
                with open(summary_file, 'a') as f:
                    f.write("Importance des features (coefficients):\n")
                    for feature, importance in top_features[:20]:
                        f.write(f"  {feature}: {importance:.4f}\n")
                    f.write("\n")
                
                # Afficher les 10 premiers dans la console
                for feature, importance in top_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
                
                # Générer un nuage de mots si on a des features
                try:
                    plt.figure(figsize=(10, 6))
                    wordcloud_fig = interpreter.plot_word_cloud(top_n=200)
                    wordcloud_path = os.path.join(args.output_dir, "feature_importance_wordcloud.png")
                    wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                    print(f"Nuage de mots sauvegardé: {wordcloud_path}")
                    
                    # Générer aussi une version avec plus de mots pour analyse détaillée
                    plt.figure(figsize=(12, 8))
                    detailed_wordcloud = interpreter.plot_word_cloud(top_n=500)
                    detailed_path = os.path.join(args.output_dir, "feature_importance_wordcloud_detailed.png")
                    detailed_wordcloud.savefig(detailed_path, dpi=300, bbox_inches='tight')
                    print(f"Nuage de mots détaillé sauvegardé: {detailed_path}")
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
            # Limiter à un sous-ensemble pour accélérer (par exemple, 30% des données)
            sample_size = min(1000, int(X.shape[0] * 0.3))
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            y_sample = [labels[i] for i in indices]
            
            print(f"Running permutation importance on {sample_size} samples...")
            perm_features = interpreter.permutation_feature_importance(X_sample, y_sample, n_repeats=5)
            
            if perm_features:
                # Enregistrer dans le fichier de résumé
                with open(summary_file, 'a') as f:
                    f.write("Importance des features (permutation):\n")
                    for feature, importance in perm_features[:20]:
                        f.write(f"  {feature}: {importance:.4f}\n")
                    f.write("\n")
                
                # Afficher les 10 premiers dans la console
                for feature, importance in perm_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
                
                # Générer un graphique des features importantes
                plt.figure(figsize=(10, 8))
                feature_names = [f for f, _ in perm_features[:15]]
                importances = [i for _, i in perm_features[:15]]
                plt.barh(feature_names[::-1], importances[::-1])
                plt.xlabel('Importance (permutation)')
                plt.title('Top 15 features par importance de permutation')
                plt.tight_layout()
                
                perm_importance_path = os.path.join(args.output_dir, "permutation_importance.png")
                plt.savefig(perm_importance_path, dpi=300, bbox_inches='tight')
                print(f"Graphique d'importance par permutation sauvegardé: {perm_importance_path}")
                plt.show()
                
                # Comparer avec les résultats des coefficients
                if "coefficients" in args.interpretation and hasattr(classifier.model, "coef_"):
                    print("\nComparaison des méthodes d'importance:")
                    coef_features = set([f for f, _ in top_features[:20]])
                    perm_features_set = set([f for f, _ in perm_features[:20]])
                    overlap = coef_features.intersection(perm_features_set)
                    print(f"Chevauchement dans le top 20: {len(overlap)} features")
                    
                    # Enregistrer la comparaison
                    with open(summary_file, 'a') as f:
                        f.write("Comparaison des méthodes d'importance:\n")
                        f.write(f"Chevauchement dans le top 20: {len(overlap)} features\n")
                        f.write("Features communes:\n")
                        for feature in overlap:
                            f.write(f"  {feature}\n")
                        f.write("\n")
        except Exception as e:
            print(f"Erreur lors de l'analyse de l'importance par permutation: {str(e)}")
        print(f"Permutation importance analysis completed in {time.time() - start_perm:.2f} seconds")
    
    # Sauvegarde de tous les résultats d'interprétation pour le rapport
    print(f"\nRésumé d'interprétation sauvegardé: {summary_file}")
    print(f"Total interpretation time: {time.time() - start_total:.2f} seconds")

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
    
    # Créer un résumé des résultats pour le rapport
    summary_file = os.path.join(args.output_dir, "cross_validation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=== Résultats de Validation Croisée entre Datasets ===\n\n")
        f.write(f"{'Train → Test':<20} {'Précision':<10} {'F1 Macro':<10} {'Classes':<10} {'Train':<8} {'Test':<8}\n")
        f.write("-" * 70 + "\n")
    
    # Afficher les résultats sous forme de tableau
    print("\nRésultats de la validation croisée:")
    print("-" * 80)
    print(f"{'Train → Test':<20} {'Précision':<10} {'F1 Macro':<10} {'Classes':<10} {'Train':<8} {'Test':<8}")
    print("-" * 80)
    
    # Collecter les données pour le graphique
    pairs = []
    accuracies = []
    f1_scores = []
    
    for key, result in results.items():
        if "error" in result:
            print(f"{key:<20} {result['error']}")
            with open(summary_file, 'a') as f:
                f.write(f"{key:<20} {result['error']}\n")
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
        
        # Ajouter au fichier de résumé
        with open(summary_file, 'a') as f:
            f.write(f"{display_key:<20} {accuracy:.3f}{'':<5} {f1:.3f}{'':<5} {common_classes:<10} {train_size:<8} {test_size:<8}\n")
        
        # Collecter pour le graphique
        pairs.append(display_key)
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    print("-" * 80)
    with open(summary_file, 'a') as f:
        f.write("-" * 70 + "\n")
    
    # Créer un graphique comparatif des résultats de cross-validation
    if pairs:
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(pairs))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Précision', color='skyblue')
        plt.bar(x + width/2, f1_scores, width, label='F1 Macro', color='lightgreen')
        
        plt.xlabel('Paires de datasets')
        plt.ylabel('Score')
        plt.title('Résultats de la validation croisée entre datasets')
        plt.xticks(x, pairs, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarder le graphique
        cv_results_path = os.path.join(args.output_dir, "cross_validation_results.png")
        plt.savefig(cv_results_path, dpi=300, bbox_inches='tight')
        print(f"Graphique des résultats de validation croisée sauvegardé: {cv_results_path}")
        plt.show()
    
    # Sauvegarder les résultats
    output_file = os.path.join(args.output_dir, "cross_validation_results.npy")
    np.save(output_file, results)
    print(f"Résultats sauvegardés: {output_file}")
    print(f"Résumé des résultats sauvegardé: {summary_file}")

if __name__ == "__main__":
    main() 
