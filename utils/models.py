import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TextClassifier:
    """
    Classe pour entraîner et évaluer des modèles de classification sur des textes vectorisés
    """
    def __init__(self, model_type: str = "logistic", **kwargs):
        """
        Initialise le classifieur
        
        Args:
            model_type: Type de modèle ('logistic' pour régression logistique)
            **kwargs: Paramètres supplémentaires pour le modèle
        """
        self.model_type = model_type.lower()
        self.model = None
        
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                C=kwargs.get("C", 1.0),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1)
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    def train(self, X: np.ndarray, y: List[str], test_size: float = 0.2, 
             random_state: int = 42, stratify: bool = True) -> Dict[str, Any]:
        """
        Entraîne le modèle et évalue ses performances
        
        Args:
            X: Matrice de features (documents vectorisés)
            y: Liste des labels
            test_size: Proportion du jeu de test
            random_state: Graine aléatoire pour la reproductibilité
            stratify: Si True, stratifie l'échantillonnage par label
            
        Returns:
            Dictionnaire des résultats d'évaluation
        """
        # Diviser les données en train/test
        stratify_data = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
        )
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        
        # Évaluer sur le jeu de test
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Créer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_test": y_test
        }
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Prédit les labels pour de nouvelles données
        
        Args:
            X: Matrice de features (documents vectorisés)
            
        Returns:
            Liste des prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez train() d'abord.")
        
        return self.model.predict(X)

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                         title: str = "Matrice de confusion", 
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Affiche la matrice de confusion
    
    Args:
        cm: Matrice de confusion
        labels: Liste des labels
        title: Titre du graphique
        figsize: Taille de la figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def evaluate_multiple_embeddings(embedding_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Compare les performances de différentes méthodes d'embedding
    
    Args:
        embedding_results: Dictionnaire {nom_méthode: résultats_évaluation}
    """
    methods = list(embedding_results.keys())
    accuracies = [results["accuracy"] for results in embedding_results.values()]
    
    # Afficher les performances globales
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color="skyblue")
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Méthode d\'embedding')
    plt.ylabel('Accuracy')
    plt.title('Comparaison des performances des différentes méthodes d\'embedding')
    plt.ylim(0, max(accuracies) + 0.1)
    plt.tight_layout()
    plt.show()
    
    # Afficher les F1-scores par classe pour chaque méthode
    for method, results in embedding_results.items():
        report = results["classification_report"]
        
        # Exclure les moyennes et le support
        classes = [cls for cls in report.keys() if cls not in ["accuracy", "macro avg", "weighted avg"]]
        f1_scores = [report[cls]["f1-score"] for cls in classes]
        
        # Limiter le nombre de classes à afficher si trop nombreuses
        max_classes = 15
        if len(classes) > max_classes:
            indices = np.argsort(f1_scores)[-max_classes:]
            classes = [classes[i] for i in indices]
            f1_scores = [f1_scores[i] for i in indices]
        
        plt.figure(figsize=(12, 6))
        plt.barh(classes, f1_scores, color="lightgreen")
        plt.xlabel('F1-score')
        plt.ylabel('Classe')
        plt.title(f'F1-scores par classe pour {method}')
        plt.xlim(0, 1.0)
        plt.tight_layout()
        plt.show() 
