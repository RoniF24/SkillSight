from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# -------------------------
# Helpers
# -------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0
    return p, r, f1


# -------------------------
# Core metrics
# -------------------------
def compute_metrics_onepass_flat(eval_pred) -> Dict[str, float]:
    """
    For OnePass:
      logits: [B, S, 3]
      labels: [B, S]

    We flatten everything to:
      logits_flat: [N, 3]
      labels_flat: [N]
    and compute:
      - 3class accuracy
      - any (0 vs non-zero) precision/recall/f1
      - type (1 vs 2) accuracy/f1 on GT-present only
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Flatten
    logits_f = logits.reshape(-1, 3)
    labels_f = labels.reshape(-1)

    # Pred class
    pred_f = logits_f.argmax(axis=-1)

    # 3-class accuracy
    acc_3 = float((pred_f == labels_f).mean())

    # Any (0 vs non-zero)
    gt_any = (labels_f != 0).astype(np.int32)
    pr_any = (pred_f != 0).astype(np.int32)

    tp = int(((gt_any == 1) & (pr_any == 1)).sum())
    fp = int(((gt_any == 0) & (pr_any == 1)).sum())
    fn = int(((gt_any == 1) & (pr_any == 0)).sum())
    any_p, any_r, any_f1 = _prf(tp, fp, fn)

    # Type (1 vs 2) on GT-present only
    mask_present = labels_f != 0
    gt_type = labels_f[mask_present]
    pr_type = pred_f[mask_present]

    # Only meaningful if we also predicted non-zero; but for training-time metric
    # this is a simple view: among GT-present slots, did we get class 1/2 correct?
    type_acc = float((pr_type == gt_type).mean()) if gt_type.size > 0 else 0.0

    # Type F1 treating EXPLICIT (2) as positive, IMPLICIT (1) as negative
    # (you can also swap; just be consistent in your report)
    if gt_type.size > 0:
        gt_pos = (gt_type == 2).astype(np.int32)
        pr_pos = (pr_type == 2).astype(np.int32)
        tp2 = int(((gt_pos == 1) & (pr_pos == 1)).sum())
        fp2 = int(((gt_pos == 0) & (pr_pos == 1)).sum())
        fn2 = int(((gt_pos == 1) & (pr_pos == 0)).sum())
        _, _, type_f1 = _prf(tp2, fp2, fn2)
        type_support = int(gt_type.size)
    else:
        type_f1 = 0.0
        type_support = 0

    return {
        "eval_acc_3class": acc_3,
        "eval_any_precision": any_p,
        "eval_any_recall": any_r,
        "eval_any_f1": any_f1,
        "eval_type_acc_on_gt_present": type_acc,
        "eval_type_f1_explicit_positive": float(type_f1),
        "eval_type_support": float(type_support),
    }
