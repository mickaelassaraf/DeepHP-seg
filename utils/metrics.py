import torch
from sklearn.metrics import confusion_matrix, classification_report


def compute_metrics(labels, predictions, num_classes):
    """
    Calculer les métriques détaillées : précision, rappel, F1-score.
    """
    conf_matrix = confusion_matrix(labels, predictions, labels=range(num_classes))
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    f1_scores = [report[str(i)]["f1-score"] for i in range(num_classes)]
    precision_scores = [report[str(i)]["precision"] for i in range(num_classes)]
    recall_scores = [report[str(i)]["recall"] for i in range(num_classes)]
    
    metrics = {
        "confusion_matrix": conf_matrix,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    return metrics