import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import os
import pickle

class TextClassifier:
    def __init__(self, model_type: str = "logistic", **kwargs):
        self.model_type = model_type.lower()
        self.model = None
        
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                C=kwargs.get("C", 1.0),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                class_weight='balanced'
            )
        elif self.model_type == "svm":
            self.model = SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "linear"),
                random_state=kwargs.get("random_state", 42),
                class_weight='balanced'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                random_state=kwargs.get("random_state", 42),
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X: np.ndarray, y: List[str], test_size: float = 0.2, 
             random_state: int = 42, stratify: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        # Check label distribution
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"Class distribution: {len(class_counts)} classes, min samples: {min_count}, max samples: {max_count}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        # Determine stratification
        stratify_data = None
        if stratify:
            if min_count >= 2:
                stratify_data = y
            else:
                print(f"Warning: Some classes have fewer than 2 samples (minimum: {min_count}).")
                print("Falling back to non-stratified split.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
        )
        
        print(f"Split data in {time.time() - start_time:.2f} seconds")
        cv_start = time.time()
        
        # Optional: cross-validation score with parallelization
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, n_jobs=-1)
        print(f"Cross-validation score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Cross-validation completed in {time.time() - cv_start:.2f} seconds")
        
        # Train model
        fit_start = time.time()
        self.model.fit(X_train, y_train)
        print(f"Model fitting completed in {time.time() - fit_start:.2f} seconds")
        
        # Evaluate
        eval_start = time.time()
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")
        print(f"Total training pipeline completed in {time.time() - start_time:.2f} seconds")
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_test": y_test,
            "cv_scores": cv_scores
        }
    
    def predict(self, X: np.ndarray) -> List[str]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, path: str, vectorizer=None) -> None:
        """Sauvegarde le modèle et éventuellement le vectorizeur dans un fichier pickle"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Sauvegarder tout dans un dictionnaire
        data = {
            'model': self.model,
            'model_type': self.model_type,
            'vectorizer': vectorizer
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modèle et vectorizeur sauvegardés dans: {path}")
    
    @staticmethod
    def load_model(path: str) -> Tuple[Any, Any]:
        """Charge un modèle et son vectorizeur depuis un fichier pickle"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Créer une nouvelle instance
        classifier = TextClassifier(model_type=data['model_type'])
        classifier.model = data['model']
        
        return classifier, data.get('vectorizer')

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                         title: str = "Confusion Matrix", 
                         figsize: Tuple[int, int] = (10, 8),
                         output_dir: str = "results_rapport") -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('Actual values')
    plt.title(title)
    plt.tight_layout()
    
    # Sauvegarder l'image
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"confusion_matrix_{title.lower().replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved: {output_file}")
    
    plt.show()

def evaluate_multiple_embeddings(embedding_results: Dict[str, Dict[str, Any]], 
                               output_dir: str = "results_rapport") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(embedding_results.keys())
    accuracies = [results["accuracy"] for results in embedding_results.values()]
    
    # Graphique de comparaison des précisions
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color="skyblue")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Embedding method')
    plt.ylabel('Accuracy')
    plt.title('Performance comparison of embedding methods')
    plt.ylim(0, max(accuracies) + 0.1)
    plt.tight_layout()
    
    # Sauvegarder l'image
    output_file = os.path.join(output_dir, "embedding_accuracy_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison saved: {output_file}")
    
    plt.show()
    
    # Graphiques des F1-scores par classe pour chaque méthode
    for method, results in embedding_results.items():
        report = results["classification_report"]
        
        classes = [cls for cls in report.keys() if cls not in ["accuracy", "macro avg", "weighted avg"]]
        f1_scores = [report[cls]["f1-score"] for cls in classes]
        
        max_classes = 15
        if len(classes) > max_classes:
            indices = np.argsort(f1_scores)[-max_classes:]
            classes = [classes[i] for i in indices]
            f1_scores = [f1_scores[i] for i in indices]
        
        plt.figure(figsize=(12, 6))
        plt.barh(classes, f1_scores, color="lightgreen")
        plt.xlabel('F1-score')
        plt.ylabel('Class')
        plt.title(f'F1-scores by class for {method}')
        plt.xlim(0, 1.0)
        plt.tight_layout()
        
        # Sauvegarder l'image
        output_file = os.path.join(output_dir, f"f1_scores_{method}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"F1 scores for {method} saved: {output_file}")
        
        plt.show() 
