#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
    parser.add_argument("--mode", type=str, choices=["tokenize", "classify", "all"], default="all")
    parser.add_argument("--label", type=str, choices=["artiste", "album", "genre", "année"], default="artiste")
    parser.add_argument("--vectorizers", type=str, nargs="+", 
                        choices=["bow", "tfidf", "word2vec", "fasttext", "transformer", "all"],
                        default=["all"])
    parser.add_argument("--classifier", type=str, choices=["logistic", "svm", "random_forest"],
                       default="logistic", help="Classification algorithm to use")
    parser.add_argument("--save_vectors", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--min_samples", type=int, default=1, help="Minimum samples per class")
    parser.add_argument("--bpe_merges", type=int, default=1000, help="Number of BPE merges to perform")
    parser.add_argument("--use_stopwords", action="store_true", help="Filter out stopwords during tokenization")
    parser.add_argument("--top_classes", type=int, default=0, help="Only keep top N classes with most samples (0=all)")
    parser.add_argument("--pca", type=int, default=0, help="Apply PCA to reduce dimensions (0=disabled)")
    parser.add_argument("--confusion_matrix", action="store_true", help="Plot confusion matrix for best model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    
    print("\n=== Loading data ===")
    texts, metadata_list = load_lyrics_dataset(args.input_dir)
    print(f"Total songs: {len(texts)}")
    
    labels = get_label_from_metadata(metadata_list, args.label)
    print(f"Unique {args.label}s: {len(set(labels))}")
    
    # Filter out classes with too few samples
    if args.min_samples > 1:
        from collections import Counter
        label_counts = Counter(labels)
        valid_labels = {label for label, count in label_counts.items() if count >= args.min_samples}
        
        filtered_indices = [i for i, label in enumerate(labels) if label in valid_labels]
        texts = [texts[i] for i in filtered_indices]
        metadata_list = [metadata_list[i] for i in filtered_indices]
        labels = [labels[i] for i in filtered_indices]
        
        print(f"Filtered to {len(texts)} songs with at least {args.min_samples} samples per {args.label}")
        print(f"Remaining unique {args.label}s: {len(valid_labels)}")
    
    # Optionally keep only top N classes
    if args.top_classes > 0:
        from collections import Counter
        label_counts = Counter(labels)
        top_labels = {label for label, _ in label_counts.most_common(args.top_classes)}
        
        filtered_indices = [i for i, label in enumerate(labels) if label in top_labels]
        texts = [texts[i] for i in filtered_indices]
        metadata_list = [metadata_list[i] for i in filtered_indices]
        labels = [labels[i] for i in filtered_indices]
        
        print(f"Filtered to {len(texts)} songs from top {args.top_classes} {args.label}s")
        print(f"Remaining unique {args.label}s: {len(set(labels))}")
    
    if args.mode in ["tokenize", "all"]:
        print("\n=== Tokenizing lyrics ===")
        tokenizer = BPETokenizer(
            dataset=texts, 
            num_merges=args.bpe_merges,
            use_stopwords=args.use_stopwords
        )
        tokenized_texts = [tokenizer(text) for text in texts]
        print(f"Tokenized texts: {len(tokenized_texts)}")
        save_tokenized_lyrics(tokenized_texts, metadata_list, args.output_dir)
    
    if args.mode in ["classify", "all"]:
        print("\n=== Vectorization and classification ===")
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
            print(f"\nVectorization method: {method}")
            vectorizer = TextVectorizer(method=method)
            X = vectorizer.fit_transform(texts)
            print(f"Vector shape: {X.shape}")
            
            # Optional dimensionality reduction
            if args.pca > 0 and X.shape[1] > args.pca:
                pca = PCA(n_components=args.pca)
                X_reduced = pca.fit_transform(X)
                explained_var = sum(pca.explained_variance_ratio_) * 100
                print(f"Applied PCA: {X.shape[1]} -> {args.pca} dimensions ({explained_var:.2f}% variance retained)")
                X = X_reduced
            
            if args.save_vectors:
                output_dir = "vectors"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{method}_vectors.npy")
                np.save(output_file, X)
                print(f"Vectors saved: {output_file}")
            
            classifier = TextClassifier(model_type=args.classifier)
            eval_results = classifier.train(
                X, labels, 
                test_size=0.2, 
                random_state=args.random_seed, 
                stratify=True
            )
            
            accuracy = eval_results["accuracy"]
            report = eval_results["classification_report"]
            
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Macro F1-score: {report['macro avg']['f1-score']:.3f}")
            
            results[method] = eval_results
            
            # Track best method
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
                best_X = X
        
        if len(results) > 1:
            print("\n=== Comparing vectorization methods ===")
            evaluate_multiple_embeddings(results)
            print(f"\nBest vectorization method: {best_method} (Accuracy: {best_accuracy:.3f})")
        
        # Plot confusion matrix for best model if requested
        if args.confusion_matrix and best_method:
            best_results = results[best_method]
            cm = best_results["confusion_matrix"]
            
            # Get classes that actually appeared in test set
            y_test = best_results["y_test"]
            y_pred = best_results["y_pred"]
            classes = sorted(list(set(y_test)))
            
            # If too many classes, limit to top 15 by frequency
            if len(classes) > 15:
                from collections import Counter
                class_counts = Counter(y_test)
                classes = [c for c, _ in class_counts.most_common(15)]
                
                # Filter confusion matrix to include only these classes
                indices = [i for i, c in enumerate(y_test) if c in classes]
                y_test_filtered = [y_test[i] for i in indices]
                y_pred_filtered = [y_pred[i] for i in indices]
                cm = confusion_matrix(y_test_filtered, y_pred_filtered)
            
            plot_confusion_matrix(cm, classes, title=f"Confusion Matrix for {best_method}")
            plt.savefig(f"confusion_matrix_{best_method}.png")
            print(f"Confusion matrix saved as confusion_matrix_{best_method}.png")

if __name__ == "__main__":
    main() 
