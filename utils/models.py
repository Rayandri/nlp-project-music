import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class TextClassifier:
    def __init__(self, model_type: str = "logistic", **kwargs):
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
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X: np.ndarray, y: List[str], test_size: float = 0.2, 
             random_state: int = 42, stratify: bool = True) -> Dict[str, Any]:
        stratify_data = None
        if stratify:
            # Check if all classes have at least 2 samples for stratification
            class_counts = Counter(y)
            min_count = min(class_counts.values())
            
            if min_count >= 2:
                stratify_data = y
            else:
                print(f"Warning: Some classes have fewer than 2 samples (minimum: {min_count}).")
                print("Falling back to non-stratified split.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_test": y_test
        }
    
    def predict(self, X: np.ndarray) -> List[str]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                         title: str = "Confusion Matrix", 
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('Actual values')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def evaluate_multiple_embeddings(embedding_results: Dict[str, Dict[str, Any]]) -> None:
    methods = list(embedding_results.keys())
    accuracies = [results["accuracy"] for results in embedding_results.values()]
    
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
    plt.show()
    
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
        plt.show() 
