"""codes/models/pairwise/metrics_pairwise.py

Metric helpers for the Pairwise model.

We evaluate the pairwise classifier on two complementary views:
- **Any (detection)**: binary metric where label 0 means the skill is absent, and labels {1,2} mean present.
- **Type (implicit vs explicit)**: binary metric computed only on examples where BOTH:
    (a) the ground-truth label is non-zero, and
    (b) the model predicted non-zero.
    This measures type correctness conditioned on successful detection.

The main entry point is `compute_metrics_pairwise()`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def confusion_3x3(y_true: List[int], y_pred: List[int]) -> List[List[int]]:
    cm = [[0, 0, 0] for _ in range(3)]
    for t, p in zip(y_true, y_pred):
        if t not in (0, 1, 2) or p not in (0, 1, 2):
            # keep it safe; ignore invalid labels instead of crashing
            continue
        cm[t][p] += 1
    return cm


def prf_binary(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    # Positive class is 1.
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_metrics_pairwise(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    # Any/detection: 0 vs non-zero.
    y_true_any = [0 if t == 0 else 1 for t in y_true]
    y_pred_any = [0 if p == 0 else 1 for p in y_pred]
    p_any, r_any, f1_any = prf_binary(y_true_any, y_pred_any)

    # Type: only where GT is non-zero AND prediction is non-zero.
    # This measures "explicit vs implicit" after the model has detected the skill.
    idx = [i for i, t in enumerate(y_true) if t != 0 and y_pred[i] != 0]
    y_true_type = [1 if y_true[i] == 2 else 0 for i in idx]  # implicit(1)->0, explicit(2)->1
    y_pred_type = [1 if y_pred[i] == 2 else 0 for i in idx]
    p_type, r_type, f1_type = prf_binary(y_true_type, y_pred_type) if idx else (0.0, 0.0, 0.0)

    cm = confusion_3x3(y_true, y_pred)

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0

    return {
        "acc_3class": acc,
        "any_precision": p_any,
        "any_recall": r_any,
        "any_f1": f1_any,
        "type_precision": p_type,
        "type_recall": r_type,
        "type_f1": f1_type,
        "type_support": float(len(idx)),  # helps interpret Type (how many samples it was computed on)
        "cm00": cm[0][0], "cm01": cm[0][1], "cm02": cm[0][2],
        "cm10": cm[1][0], "cm11": cm[1][1], "cm12": cm[1][2],
        "cm20": cm[2][0], "cm21": cm[2][1], "cm22": cm[2][2],
    }