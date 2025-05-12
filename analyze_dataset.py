#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze class distribution and statistics in the dataset.
"""

import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

from utils.data_loader import load_lyrics_dataset, get_label_from_metadata

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyser la distribution des classes dans le dataset")
    parser.add_argument("--input_dir", type=str, default="lyrics_dataset")
    parser.add_argument("--label", type=str, choices=["artiste", "album", "genre", "année"], default="artiste")
    parser.add_argument("--min_samples", type=int, default=0, help="Minimum samples per class to include in analysis")
    parser.add_argument("--top_n", type=int, default=20, help="Show only top N classes in plots")
    parser.add_argument("--output", type=str, default="dataset_analysis.png", help="Output file for plots")
    return parser.parse_args()

def analyze_distribution(labels, title, min_samples=0, top_n=20):
    """Analyze and visualize label distribution in the dataset."""
    counter = Counter(labels)
    if min_samples > 0:
        counter = Counter({k: v for k, v in counter.items() if v >= min_samples})
    most_common = counter.most_common(top_n)
    class_counts = list(counter.values())
    total_classes = len(counter)
    min_count = min(class_counts)
    max_count = max(class_counts)
    avg_count = sum(class_counts) / total_classes
    median_count = sorted(class_counts)[total_classes // 2]
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n=== Distribution Analysis: {title} ===")
    print(f"Total classes: {total_classes}")
    print(f"Total samples: {sum(class_counts)}")
    print(f"Samples per class: min={min_count}, max={max_count}, avg={avg_count:.1f}, median={median_count}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    thresholds = [2, 5, 10, 20, 50]
    for t in thresholds:
        classes_with_n = sum(1 for count in class_counts if count >= t)
        print(f"Classes with at least {t} samples: {classes_with_n} ({100*classes_with_n/total_classes:.1f}%)")
    plt.figure(figsize=(12, 8))
    classes, counts = zip(*most_common)
    plt.bar(range(len(classes)), counts)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.title(f"Top {len(classes)} classes by frequency - {title}")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    return plt.gcf()

def main():
    """Main function for dataset analysis."""
    args = parse_args()
    print(f"Loading data from {args.input_dir}...")
    texts, metadata_list = load_lyrics_dataset(args.input_dir)
    print(f"Loaded {len(texts)} songs")
    labels = get_label_from_metadata(metadata_list, args.label)
    fig = analyze_distribution(
        labels, 
        title=f"Distribution of {args.label}",
        min_samples=args.min_samples,
        top_n=args.top_n
    )
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Analysis plot saved to {args.output}")
    print("\nTop 10 most common classes:")
    for label, count in Counter(labels).most_common(10):
        print(f"  {label}: {count} samples")
    singleton_classes = [label for label, count in Counter(labels).items() if count == 1]
    print(f"\nClasses with only one sample: {len(singleton_classes)}")
    if len(singleton_classes) <= 10:
        for cls in singleton_classes:
            print(f"  {cls}")
    else:
        print(f"  {', '.join(singleton_classes[:10])}... (and {len(singleton_classes)-10} more)")

if __name__ == "__main__":
    main() 
