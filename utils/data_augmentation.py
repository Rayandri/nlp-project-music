#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import wordnet
import re
import copy
from collections import Counter

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('french'))
except:
    STOPWORDS = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
                    'ce', 'cette', 'ces', 'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 'a', 'au', 'aux',
                    'de', 'du', 'des', 'en', 'par', 'pour', 'avec', 'sans', 'sous', 'sur', 'dans', 'entre', 'vers', 'je', 'tu'])

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

class DataAugmenter:
    """Text data augmenter supporting various methods."""
    def __init__(self):
        pass
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        words = text.split()
        if len(words) == 1:
            return text
        keep_prob = 1 - p
        new_words = []
        for word in words:
            if random.random() < keep_prob:
                new_words.append(word)
        if len(new_words) == 0:
            return random.choice(words)
        return ' '.join(new_words)
    def random_swap(self, text: str, n: int = 1) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    def random_insertion(self, text: str, n: int = 1) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            word_to_insert = random.choice(words)
            insert_position = random.randint(0, len(words))
            words.insert(insert_position, word_to_insert)
        return ' '.join(words)
    def synonym_replacement(self, text: str, n: int = 1, language: str = 'fra') -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        non_stopwords = [word for word in words if word.lower() not in STOPWORDS]
        if len(non_stopwords) == 0:
            return text
        n = min(n, len(non_stopwords))
        words_to_replace = random.sample(non_stopwords, n)
        new_words = []
        for word in words:
            if word in words_to_replace:
                synonym = self._get_synonym(word, language)
                if synonym:
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        return ' '.join(new_words)
    def _get_synonym(self, word: str, language: str = 'fra') -> Optional[str]:
        synonyms = []
        for syn in wordnet.synsets(word, lang=language):
            for lemma in syn.lemmas(language):
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        if not synonyms:
            return None
        return random.choice(synonyms)
    def back_translation(self, text: str, from_lang: str = 'fr', to_lang: str = 'en') -> str:
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.2:
                word = words[i]
                if len(word) > 3:
                    if random.random() < 0.5:
                        pos = random.randint(1, len(word)-2)
                        words[i] = word[:pos] + word[pos+1:]
                    else:
                        pos = random.randint(1, len(word)-1)
                        words[i] = word[:pos] + word[pos] + word[pos:]
        return ' '.join(words)
    def contextual_augmentation(self, text: str) -> str:
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.15 and len(words[i]) > 3:
                words[i] = self._simulate_contextual_replacement(words[i])
        return ' '.join(words)
    def _simulate_contextual_replacement(self, word: str) -> str:
        transformations = [
            lambda w: w.upper() if w.islower() else w.lower(),
            lambda w: w + "s" if not w.endswith("s") else w[:-1],
            lambda w: w + "e" if not w.endswith("e") else w[:-1],
            lambda w: w.replace("a", "e") if "a" in w else w.replace("e", "a") if "e" in w else w,
            lambda w: w
        ]
        return random.choice(transformations)(word)
    def augment(self, text: str, methods: List[str] = None, num_augmentations: int = 1) -> List[str]:
        if methods is None:
            methods = ["random_deletion", "random_swap", "random_insertion", 
                       "synonym_replacement", "back_translation", "contextual_augmentation"]
        augmented_texts = []
        for _ in range(num_augmentations):
            method = random.choice(methods)
            if method == "random_deletion":
                augmented_texts.append(self.random_deletion(text))
            elif method == "random_swap":
                augmented_texts.append(self.random_swap(text))
            elif method == "random_insertion":
                augmented_texts.append(self.random_insertion(text))
            elif method == "synonym_replacement":
                augmented_texts.append(self.synonym_replacement(text))
            elif method == "back_translation":
                augmented_texts.append(self.back_translation(text))
            elif method == "contextual_augmentation":
                augmented_texts.append(self.contextual_augmentation(text))
        return augmented_texts

def augment_dataset(texts: List[str], labels: List[str], methods: List[str] = None, 
                   factor: float = 0.5, balanced: bool = True) -> Tuple[List[str], List[str]]:
    augmenter = DataAugmenter()
    augmented_texts = []
    augmented_labels = []
    num_new_examples = int(len(texts) * factor)
    if balanced:
        label_counts = Counter(labels)
        label_indices = {label: [i for i, l in enumerate(labels) if l == label] for label in set(labels)}
        examples_per_class = {}
        for label in label_counts:
            inverse_freq = 1 / label_counts[label]
            examples_per_class[label] = int(num_new_examples * inverse_freq / sum(1/count for count in label_counts.values()))
        for label, num_examples in examples_per_class.items():
            indices = label_indices[label]
            num_examples = min(num_examples, len(indices) * 3)
            for _ in range(num_examples):
                idx = random.choice(indices)
                text = texts[idx]
                augmented = augmenter.augment(text, methods, num_augmentations=1)[0]
                augmented_texts.append(augmented)
                augmented_labels.append(label)
    else:
        for _ in range(num_new_examples):
            idx = random.randint(0, len(texts) - 1)
            text = texts[idx]
            label = labels[idx]
            augmented = augmenter.augment(text, methods, num_augmentations=1)[0]
            augmented_texts.append(augmented)
            augmented_labels.append(label)
    all_texts = texts + augmented_texts
    all_labels = labels + augmented_labels
    return all_texts, all_labels

def evaluate_augmentation_impact(texts: List[str], labels: List[str], 
                               augmentation_methods: List[str] = None,
                               classifier_type: str = "logistic"):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    results = {}
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    baseline_f1 = f1_score(y_test, y_pred, average='macro')
    results["baseline"] = {
        "accuracy": baseline_accuracy,
        "f1_score": baseline_f1,
        "train_size": len(X_train)
    }
    for factor in [0.3, 0.5, 1.0]:
        X_train_aug, y_train_aug = augment_dataset(X_train, y_train, methods=augmentation_methods, factor=factor)
        vectorizer = TfidfVectorizer()
        X_train_aug_vec = vectorizer.fit_transform(X_train_aug)
        X_test_vec = vectorizer.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_aug_vec, y_train_aug)
        y_pred = clf.predict(X_test_vec)
        aug_accuracy = accuracy_score(y_test, y_pred)
        aug_f1 = f1_score(y_test, y_pred, average='macro')
        results[f"augmented_{factor}"] = {
            "accuracy": aug_accuracy,
            "f1_score": aug_f1,
            "train_size": len(X_train_aug),
            "improvement": aug_accuracy - baseline_accuracy
        }
    return results 
