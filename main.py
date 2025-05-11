#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from typing import List, Dict

from utils.tokenizer import BPETokenizer
from utils.data_loader import load_lyrics_dataset, save_tokenized_lyrics, get_label_from_metadata
from utils.vectorizers import TextVectorizer
from utils.models import TextClassifier, evaluate_multiple_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Lyrics classification")
    
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset")
    parser.add_argument("--output_dir", type=str, default="tokenized_lyrics_dataset")
    parser.add_argument("--mode", type=str, choices=["tokenize", "classify", "all"], default="all")
    parser.add_argument("--label", type=str, choices=["artiste", "album", "genre", "année"], default="artiste")
    parser.add_argument("--vectorizers", type=str, nargs="+", 
                        choices=["bow", "tfidf", "word2vec", "fasttext", "transformer", "all"],
                        default=["all"])
    parser.add_argument("--save_vectors", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n=== Loading data ===")
    texts, metadata_list = load_lyrics_dataset(args.input_dir)
    print(f"Total songs: {len(texts)}")
    
    labels = get_label_from_metadata(metadata_list, args.label)
    print(f"Unique {args.label}s: {len(set(labels))}")
    
    if args.mode in ["tokenize", "all"]:
        print("\n=== Tokenizing lyrics ===")
        tokenizer = BPETokenizer(dataset=texts)
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
        
        for method in vectorization_methods:
            print(f"\nVectorization method: {method}")
            vectorizer = TextVectorizer(method=method)
            X = vectorizer.fit_transform(texts)
            print(f"Vector shape: {X.shape}")
            
            if args.save_vectors:
                output_dir = "vectors"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{method}_vectors.npy")
                np.save(output_file, X)
                print(f"Vectors saved: {output_file}")
            
            classifier = TextClassifier(model_type="logistic")
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
        
        if len(results) > 1:
            print("\n=== Comparing vectorization methods ===")
            evaluate_multiple_embeddings(results)

if __name__ == "__main__":
    main() 
