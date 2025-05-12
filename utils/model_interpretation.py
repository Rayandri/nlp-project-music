#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lime.lime_text import LimeTextExplainer
from sklearn.inspection import permutation_importance
import shap
import eli5
from eli5.sklearn import PermutationImportance
from collections import defaultdict
from wordcloud import WordCloud

class ModelInterpreter:
    """Classe pour interpréter les modèles de classification de texte"""
    
    def __init__(self, model, vectorizer=None):
        """Initialise l'interpréteur de modèle"""
        self.model = model
        self.vectorizer = vectorizer
        
    def get_feature_importance(self, method: str = "coefficients"):
        """Récupère l'importance des features selon différentes méthodes"""
        if method == "coefficients" and hasattr(self.model, "coef_"):
            return self._get_coefficient_importance()
        elif method == "permutation":
            raise ValueError("La méthode 'permutation' nécessite des données X et y")
        elif method == "shap":
            raise ValueError("La méthode 'shap' nécessite des données X")
        else:
            raise ValueError(f"Méthode d'importance non supportée: {method}")
            
    def _get_coefficient_importance(self):
        """Récupère l'importance des features basée sur les coefficients"""
        if not hasattr(self.model, "coef_"):
            raise ValueError("Le modèle n'a pas d'attribut 'coef_'")
            
        if not self.vectorizer:
            raise ValueError("Vectorizer non fourni, impossible d'associer les features")
            
        try:
            # Récupérer les coefficients
            coefs = self.model.coef_
            
            # Obtenir les noms des features
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Pour la classification multiclasse
            if coefs.shape[0] > 1:
                # Moyenne des valeurs absolues des coefficients pour chaque feature
                importances = np.abs(coefs).mean(axis=0)
            else:
                importances = np.abs(coefs[0])
                
            # Vérifier les dimensions
            if len(feature_names) != len(importances):
                print(f"Attention: Nombre de features ({len(feature_names)}) différent du nombre de coefficients ({len(importances)})")
                # Utiliser les dimensions disponibles
                min_len = min(len(feature_names), len(importances))
                feature_names = feature_names[:min_len]
                importances = importances[:min_len]
                
            # Créer un dictionnaire feature -> importance
            feature_importance = dict(zip(feature_names, importances))
            
            # Trier par importance décroissante
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_importance
        except Exception as e:
            print(f"Erreur lors de l'extraction de l'importance des features: {str(e)}")
            # Retourner une liste vide si erreur
            return []
    
    def explain_prediction(self, text: str, num_features: int = 10):
        """Explique la prédiction pour un texte donné"""
        if not self.vectorizer:
            raise ValueError("Vectorizer non fourni, impossible d'expliquer la prédiction")
            
        # Vectoriser le texte
        X = self.vectorizer.transform([text])
        
        # Prédire
        prediction = self.model.predict(X)[0]
        
        # Pour les modèles avec coefficients (ex. régression logistique)
        if hasattr(self.model, "coef_"):
            return self._explain_linear_prediction(text, prediction, num_features)
        # Pour les modèles basés sur les arbres
        elif hasattr(self.model, "feature_importances_"):
            return self._explain_tree_prediction(text, prediction, num_features)
        else:
            return {"prediction": prediction, "explanation": "Modèle non supporté pour l'explication"}
    
    def _explain_linear_prediction(self, text: str, prediction: Any, num_features: int = 10):
        """Explique la prédiction d'un modèle linéaire"""
        # Vectoriser le texte
        X = self.vectorizer.transform([text])
        
        # Récupérer les noms des features
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Récupérer les coefficients
        if len(self.model.classes_) == 2:
            # Cas binaire
            coefs = self.model.coef_[0]
            
            # Calculer la contribution de chaque token
            feature_values = X.toarray()[0]
            contributions = coefs * feature_values
            
            # Trier par contribution absolue
            indices = np.argsort(np.abs(contributions))[::-1]
            top_indices = indices[:num_features]
            
            # Collecter les top features et leurs contributions
            top_features = [(feature_names[i], contributions[i]) for i in top_indices]
            
            return {
                "prediction": prediction,
                "top_features": top_features,
                "positive_contribution": sum(c for _, c in top_features if c > 0),
                "negative_contribution": sum(c for _, c in top_features if c < 0)
            }
        else:
            # Cas multiclasse
            class_idx = np.where(self.model.classes_ == prediction)[0][0]
            coefs = self.model.coef_[class_idx]
            
            # Calculer la contribution de chaque token
            feature_values = X.toarray()[0]
            contributions = coefs * feature_values
            
            # Trier par contribution absolue
            indices = np.argsort(np.abs(contributions))[::-1]
            top_indices = indices[:num_features]
            
            # Collecter les top features et leurs contributions
            top_features = [(feature_names[i], contributions[i]) for i in top_indices]
            
            return {
                "prediction": prediction,
                "class_index": class_idx,
                "top_features": top_features,
                "positive_contribution": sum(c for _, c in top_features if c > 0),
                "negative_contribution": sum(c for _, c in top_features if c < 0)
            }
    
    def _explain_tree_prediction(self, text: str, prediction: Any, num_features: int = 10):
        """Explique la prédiction d'un modèle basé sur les arbres"""
        # Vectoriser le texte
        X = self.vectorizer.transform([text])
        
        # Récupérer les noms des features
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Récupérer l'importance des features
        importances = self.model.feature_importances_
        
        # Calculer l'importance de chaque token pour ce texte spécifique
        feature_values = X.toarray()[0]
        token_importances = importances * feature_values
        
        # Trier par importance
        indices = np.argsort(token_importances)[::-1]
        top_indices = indices[:num_features]
        
        # Collecter les top features et leurs importances
        top_features = [(feature_names[i], token_importances[i]) for i in top_indices]
        
        return {
            "prediction": prediction,
            "top_features": top_features,
            "total_importance": sum(i for _, i in top_features)
        }
    
    def explain_with_lime(self, text: str, num_features: int = 10, num_samples: int = 1000):
        """Explique la prédiction avec LIME (Local Interpretable Model-agnostic Explanations)"""
        # Créer une fonction de prédiction pour le pipeline modèle+vectorizer
        def predict_proba_fn(texts):
            vectorized_texts = self.vectorizer.transform(texts)
            return self.model.predict_proba(vectorized_texts)
        
        # Déterminer si c'est un classifieur binaire ou multiclasse
        class_names = self.model.classes_
        
        # Initialiser l'explainer LIME
        explainer = LimeTextExplainer(class_names=class_names)
        
        # Générer l'explication
        explanation = explainer.explain_instance(
            text, 
            predict_proba_fn, 
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Récupérer les top features pour la classe prédite
        vectorized_text = self.vectorizer.transform([text])
        predicted_class = self.model.predict(vectorized_text)[0]
        
        if len(class_names) == 2:
            # Pour classifieur binaire, LIME utilise la classe positive (1)
            class_idx = 1
        else:
            # Pour multiclasse, utiliser l'indice de la classe prédite
            class_idx = np.where(class_names == predicted_class)[0][0]
        
        # Récupérer les explications pour la classe prédite
        explanation_list = explanation.as_list(label=class_idx)
        
        return {
            "prediction": predicted_class,
            "explanations": explanation_list,
            "intercept": explanation.intercept[class_idx],
            "score": explanation.score
        }
    
    def explain_with_shap(self, X, num_samples: int = 100):
        """Explique le modèle avec SHAP (SHapley Additive exPlanations)"""
        # Créer un explainer SHAP
        if hasattr(self.model, "coef_"):
            # Pour les modèles linéaires
            explainer = shap.LinearExplainer(self.model, X[:num_samples])
        else:
            # Pour les autres modèles (comme RandomForest)
            explainer = shap.KernelExplainer(self.model.predict, X[:num_samples].toarray())
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X[:num_samples].toarray())
        
        # Préparer les résultats
        if isinstance(shap_values, list):
            # Multiclasse: une matrice de valeurs SHAP par classe
            results = []
            for class_idx, values in enumerate(shap_values):
                class_name = self.model.classes_[class_idx]
                class_result = {
                    "class": class_name,
                    "shap_values": values,
                    "mean_abs_shap": np.abs(values).mean(axis=0)
                }
                results.append(class_result)
            return results
        else:
            # Binaire: une seule matrice de valeurs SHAP
            return {
                "shap_values": shap_values,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0)
            }

    def plot_feature_importance(self, top_n: int = 20):
        """Trace un graphique des features les plus importantes"""
        feature_importance = self.get_feature_importance()
        
        # Limiter aux top_n features
        top_features = feature_importance[:top_n]
        
        # Séparer les noms et les valeurs
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Inverser l'ordre pour l'affichage
        feature_names = feature_names[::-1]
        importances = importances[::-1]
        
        # Créer le graphique
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_names)), importances)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} des features les plus importantes')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_word_cloud(self, top_n: int = 200):
        """Génère un nuage de mots basé sur l'importance des features"""
        feature_importance = self.get_feature_importance()
        
        # Limiter aux top_n features
        top_features = feature_importance[:top_n]
        
        # Créer un dictionnaire mot -> importance
        word_importance = {f[0]: f[1] for f in top_features}
        
        # Générer le nuage de mots
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              max_words=100,
                              colormap='viridis').generate_from_frequencies(word_importance)
        
        # Afficher le nuage de mots
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Nuage de mots selon l'importance des features")
        plt.tight_layout()
        
        return plt.gcf()
    
    def permutation_feature_importance(self, X, y, n_repeats: int = 10, random_state: int = 42):
        """Calcule l'importance des features par permutation"""
        # Calculer l'importance par permutation
        perm_importance = permutation_importance(
            self.model, X, y, 
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Récupérer les moyennes
        importances = perm_importance.importances_mean
        
        # Associer aux noms des features
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Créer un dictionnaire trié
        feature_importance = dict(zip(feature_names, importances))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_importance
    
    def compare_feature_importance_methods(self, X, y, n_repeats: int = 5):
        """Compare différentes méthodes d'importance des features"""
        results = {}
        
        # Méthode des coefficients (si applicable)
        if hasattr(self.model, "coef_"):
            coef_importance = self.get_feature_importance(method="coefficients")
            results["coefficients"] = coef_importance
        
        # Méthode de permutation
        perm_importance = self.permutation_feature_importance(X, y, n_repeats=n_repeats)
        results["permutation"] = perm_importance
        
        # Comparer les top features entre les méthodes
        comparison = self._compare_top_features(results)
        
        return results, comparison
    
    def _compare_top_features(self, importance_results, top_n: int = 20):
        """Compare les top features entre différentes méthodes"""
        top_features = {}
        
        # Extraire les top_n features de chaque méthode
        for method, importances in importance_results.items():
            top_features[method] = [feature for feature, _ in importances[:top_n]]
        
        # Calculer le chevauchement entre les méthodes
        overlap = {}
        methods = list(top_features.keys())
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                
                set1 = set(top_features[method1])
                set2 = set(top_features[method2])
                
                intersection = set1.intersection(set2)
                
                overlap[f"{method1}_vs_{method2}"] = {
                    "overlap_count": len(intersection),
                    "overlap_percent": len(intersection) / top_n * 100,
                    "common_features": sorted(list(intersection))
                }
        
        return overlap 
