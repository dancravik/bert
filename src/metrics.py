from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, label_names):
    """Считает по максимуму метрик классификации:
    - accuracy
    - precision / recall / f1 (macro, micro, weighted)
    - precision / recall / f1 по каждому классу отдельно
    - Matthews correlation coefficient
    - Cohen's kappa
    - log loss
    - ROC-AUC (one-vs-rest, macro)
    - confusion matrix
    Для несбалансированных многоклассовых задач (а тут классы неравномерны:
    Positive/Negative встречаются чаще Extremely-*) macro-F1 и MCC обычно
    информативнее голого accuracy, поэтому используем macro-F1 как основную
    метрику при выборе лучшей модели/конфигурации."""
    labels_idx = list(range(len(label_names)))
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for avg in ("macro", "micro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0, labels=labels_idx
        )
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)

    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels_idx))
    except ValueError:
        metrics["log_loss"] = None

    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels_idx)
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None

    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels_idx
    )
    metrics["per_class"] = {
        name: {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(label_names)
    }

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels_idx).tolist()

    return metrics
