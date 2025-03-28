import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, average_precision_score

__all__ = ["calculate_metrics"]


def calculate_metrics(output, target, threshold=None, verbose=False, simplify=True, expected_sensitivity=None):
    """
    output: logits (N,)
    target: labels (N,)
    """

    if min(output) < 0 or max(output) > 1:
        probabilities = 1 / (1 + np.exp(-output))
    else:
        probabilities = output

    auc = roc_auc_score(target, probabilities)
    aupr = average_precision_score(target, probabilities)

    
    fpr, tpr, thresholds = roc_curve(target, probabilities)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds), reverse=True)
    best_threshold = j_ordered[0][1]

    
    if expected_sensitivity is not None:
        target_tpr = expected_sensitivity
        idx = np.where(tpr >= target_tpr)[0]
        best_threshold = thresholds[idx[0]]

    
    if threshold is not None:
        best_threshold = threshold
    predictions = probabilities >= best_threshold

    
    tn, fp, fn, tp = confusion_matrix(target, predictions).ravel()
    if verbose:
        print(f"Confusion matrix: tn={tn}, fp={fp}, fn={fn}, tp={tp}")

    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    if simplify:
        return {
            "AUC": auc,
            # "AUPR": aupr,
            "ACC": accuracy,
            "SEN": sensitivity,
            "SPE": specificity,
            "PPV": ppv,
            "NPV": npv,
            "Thr": best_threshold,
        }
    
    return {
        "AUC": auc,
        # "AUPR": aupr,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Threshold": best_threshold,
    }