#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
from gensim.models import Word2Vec, FastText
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

class TextGenerator:
    def __init__(self, generator_type: str = "ngram", **kwargs):
        self.generator_type = generator_type.lower()
        self.model = None
        self.vocab = None
        self.seq_length = kwargs.get("seq_length", 10)
        self.temperature = kwargs.get("temperature", 1.0)
        self.max_length = kwargs.get("max_length", 100)
        
    def fit(self, texts: List[str], tokenized_texts: Optional[List[List[str]]] = None) -> None:
        """Entraîne le modèle de génération sur les textes donnés"""
        if tokenized_texts is None:
            tokenized_texts = [text.split() for text in texts]
            
        if self.generator_type == "ngram":
            self._train_ngram(tokenized_texts)
        elif self.generator_type == "word2vec":
            self._train_word2vec(tokenized_texts)
        elif self.generator_type == "fasttext":
            self._train_fasttext(tokenized_texts)
        elif self.generator_type == "transformer":
            self._train_transformer(texts)
        else:
            raise ValueError(f"Type de générateur non supporté: {self.generator_type}")
    
    def _train_ngram(self, tokenized_texts: List[List[str]]) -> None:
        """Implémente un modèle n-gram pour la génération de texte"""
        n = 3  # Tri-gram par défaut
        # Construction du modèle n-gram
        self.model = defaultdict(Counter)
        self.vocab = Counter()
        
        # Création des statistiques n-gram
        for tokens in tokenized_texts:
            for i in range(len(tokens) - n + 1):
                prefix = tuple(tokens[i:i+n-1])
                next_token = tokens[i+n-1]
                self.model[prefix][next_token] += 1
                self.vocab[next_token] += 1

    def _train_word2vec(self, tokenized_texts: List[List[str]]) -> None:
        """Utilise Word2Vec pour la génération basée sur les similarités"""
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        self.vocab = Counter()
        for text in tokenized_texts:
            self.vocab.update(text)
    
    def _train_fasttext(self, tokenized_texts: List[List[str]]) -> None:
        """Utilise FastText pour la génération basée sur les similarités"""
        self.model = FastText(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        self.vocab = Counter()
        for text in tokenized_texts:
            self.vocab.update(text)
    
    def _train_transformer(self, texts: List[str]) -> None:
        """Fine-tune un modèle transformer pour la génération"""
        # Utilisation de GPT-2 ou d'un autre modèle pré-entraîné
        # Note: Un vrai fine-tuning nécessiterait plus de code et de ressources
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    def generate(self, prompt: str = "", max_length: int = None) -> str:
        """Génère du texte à partir d'un prompt donné"""
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")
        
        if max_length is None:
            max_length = self.max_length
            
        if self.generator_type == "ngram":
            return self._generate_ngram(prompt, max_length)
        elif self.generator_type in ["word2vec", "fasttext"]:
            return self._generate_word_embedding(prompt, max_length)
        elif self.generator_type == "transformer":
            return self._generate_transformer(prompt, max_length)
        else:
            raise ValueError(f"Type de générateur non supporté: {self.generator_type}")
    
    def _generate_ngram(self, prompt: str, max_length: int) -> str:
        """Génère du texte avec le modèle n-gram"""
        tokens = prompt.split() if prompt else ["<START>"]
        n = 3  # Tri-gram

        # Générer du texte
        for _ in range(max_length):
            # Prendre les n-1 derniers tokens comme préfixe
            prefix = tuple(tokens[-(n-1):]) if len(tokens) >= n-1 else tuple(tokens + ["<START>"] * (n-1 - len(tokens)))
            
            # Si ce préfixe n'existe pas, en choisir un aléatoirement
            if prefix not in self.model or not self.model[prefix]:
                if not self.vocab:
                    break
                next_token = random.choice(list(self.vocab.keys()))
            else:
                # Choix pondéré du token suivant
                candidates = self.model[prefix]
                next_token = random.choices(
                    list(candidates.keys()),
                    weights=list(candidates.values()),
                    k=1
                )[0]
            
            tokens.append(next_token)
            
            # Arrêter si on a généré un token de fin
            if next_token == "<END>":
                break
                
        return " ".join(tokens)
    
    def _generate_word_embedding(self, prompt: str, max_length: int) -> str:
        """Génère du texte en utilisant les similarités de word embeddings"""
        tokens = prompt.split() if prompt else ["<START>"]
        
        # Génération basée sur les vecteurs d'embeddings
        for _ in range(max_length):
            if not tokens or tokens[-1] not in self.model.wv:
                if not self.vocab:
                    break
                # Choisir un mot au hasard dans le vocabulaire
                next_token = random.choice(list(self.vocab.keys()))
            else:
                # Trouver les mots similaires au dernier token
                similar_words = self.model.wv.most_similar(tokens[-1], topn=10)
                if not similar_words:
                    next_token = random.choice(list(self.vocab.keys()))
                else:
                    # Introduire de l'aléatoire avec le paramètre de température
                    weights = [np.exp(s[1] / self.temperature) for s in similar_words]
                    next_token = random.choices(
                        [s[0] for s in similar_words],
                        weights=weights,
                        k=1
                    )[0]
            
            tokens.append(next_token)
        
        return " ".join(tokens)
    
    def _generate_transformer(self, prompt: str, max_length: int) -> str:
        """Génère du texte en utilisant un modèle transformer"""
        if not prompt:
            prompt = "<START>"
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Générer avec le transformer
        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            temperature=self.temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    def evaluate(self, test_texts: List[str], metric: str = "perplexity") -> float:
        """Évalue le modèle de génération sur les textes de test"""
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")
            
        if metric == "perplexity":
            return self._calculate_perplexity(test_texts)
        else:
            raise ValueError(f"Métrique d'évaluation non supportée: {metric}")
    
    def _calculate_perplexity(self, test_texts: List[str]) -> float:
        """Calcule la perplexité du modèle sur les textes de test"""
        if self.generator_type == "ngram":
            # Pour n-gram, calculer la perplexité
            n = 3  # Tri-gram par défaut
            log_sum = 0
            token_count = 0
            
            for text in test_texts:
                tokens = text.split()
                for i in range(len(tokens) - n + 1):
                    prefix = tuple(tokens[i:i+n-1])
                    next_token = tokens[i+n-1]
                    
                    if prefix in self.model and next_token in self.model[prefix]:
                        probability = self.model[prefix][next_token] / sum(self.model[prefix].values())
                        log_sum -= np.log(probability)
                    else:
                        # Si la séquence n'est pas dans le modèle, utiliser un lissage
                        log_sum -= np.log(1e-10)  # Très petite probabilité
                    
                    token_count += 1
            
            if token_count == 0:
                return float('inf')
                
            perplexity = np.exp(log_sum / token_count)
            return perplexity
        elif self.generator_type == "transformer":
            # Pour les transformers, utiliser la loss comme approximation
            # Cette implémentation est simplifiée
            return 0.0  # Placeholder
        else:
            # Autres modèles ne sont pas facilement évaluables avec la perplexité
            return 0.0  # Placeholder

def benchmark_generation_models(texts: List[str], generator_types: List[str] = None):
    """Compare différents modèles de génération"""
    if generator_types is None:
        generator_types = ["ngram", "word2vec", "fasttext", "transformer"]
        
    # Séparer les données d'entraînement et de test
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    test_texts = texts[train_size:]
    
    results = {}
    generated_samples = {}
    
    for generator_type in generator_types:
        print(f"\nEntraînement du générateur: {generator_type}")
        try:
            generator = TextGenerator(generator_type=generator_type)
            generator.fit(train_texts)
            
            # Générer quelques exemples
            samples = []
            for _ in range(3):
                sample = generator.generate(max_length=50)
                samples.append(sample)
            
            generated_samples[generator_type] = samples
            
            # Évaluer le modèle
            try:
                perplexity = generator.evaluate(test_texts)
                results[generator_type] = {
                    "perplexity": perplexity
                }
                print(f"Perplexité: {perplexity:.2f}")
            except Exception as e:
                print(f"Erreur lors de l'évaluation: {str(e)}")
                results[generator_type] = {
                    "perplexity": float('inf')
                }
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
    
    return results, generated_samples 
