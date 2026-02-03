from __future__ import annotations

"""codes/models/compare_results/compare_models.py

Utility script to compare evaluation outputs across model families (Pairwise vs OnePass).

What it does:
- Finds the newest `mean_metrics_<split>.json` files under `trained_models/`.
- Normalizes a few "common" metrics (F1/Precision/Recall) across different key names.
- Writes a summary (`summary.json`, `summary.csv`) and bar charts into `compare_results_outputs/<split>_<timestamp>/`.

Typical usage:
    python codes/models/compare_results/compare_models.py --split val
    python codes/models/compare_results/compare_models.py --split test

Optional:
    --onepass_runs <run_folder_1> <run_folder_2> ...
If omitted, all OnePass run folders under the requested split directory are included.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt


# ---------------- Repo root finder ----------------
def find_repo_root() -> Path:
    """
    Repo root = folder that contains BOTH:
      - codes/
      - trained_models/
    Works no matter where you run from.
    """
    start = Path(__file__).resolve()
    for p in [start.parent] + list(start.parents):
        if (p / "codes").exists() and (p / "trained_models").exists():
            return p
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
# Keep outputs outside `codes/` so it is easy to clean/re-run without touching source.
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
OUT_DIR = REPO_ROOT / "compare_results_outputs"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def newest_file(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching files found.")
    paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return paths[0]


def find_latest_mean_metrics(
    base_dir: Path,
    split_name: str,
    must_contain: Optional[str] = None,
) -> Path:
    """
    Searches recursively for mean_metrics_<split>.json under base_dir,
    returns the newest file by mtime.
    """
    target = f"mean_metrics_{split_name}.json"
    candidates = list(base_dir.glob(f"**/{target}"))

    if must_contain:
        candidates = [p for p in candidates if must_contain in str(p)]

    if not candidates:
        raise FileNotFoundError(f"Could not find {target} under: {base_dir}")

    return newest_file(candidates)


@dataclass
class ModelResult:
    model_key: str          # e.g. "pairwise", "onepass_seed43_ep3_baseline__VAL"
    split_name: str         # "val" or "test"
    metrics_path: Path
    metrics: Dict[str, float]


# ---------- metric helpers ----------
def _first_present(metrics: Dict[str, float], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in metrics:
            return float(metrics[k])
    return None


def get_metric_value(r: ModelResult, metric_tag: str) -> float:
    """
    Provide unified values for the "common" charts (F1/Precision/Recall)
    even though Pairwise vs OnePass use different keys.

    Pairwise file keys:
      - precision_at_k / recall_at_k / f1_at_k

    OnePass file keys:
      - typed_precision_at_k / typed_recall_at_k / typed_f1_at_k
      - (also has skills-only keys, but common charts should use *typed* to be comparable)

    metric_tag:
      - "__F1__", "__PREC__", "__REC__" -> choose best matching per model
      - otherwise -> treat as normal key and read r.metrics.get(...)
    """
    if metric_tag == "__F1__":
        v = _first_present(r.metrics, ["typed_f1_at_k", "f1_at_k"])
        return v if v is not None else 0.0

    if metric_tag == "__PREC__":
        v = _first_present(r.metrics, ["typed_precision_at_k", "precision_at_k"])
        return v if v is not None else 0.0

    if metric_tag == "__REC__":
        v = _first_present(r.metrics, ["typed_recall_at_k", "recall_at_k"])
        return v if v is not None else 0.0

    return float(r.metrics.get(metric_tag, 0.0))


def bar_compare_flexible(results: List[ModelResult], metric_tag: str, title: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)

    labels = [r.model_key for r in results]
    values = [get_metric_value(r, metric_tag) for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_csv(results: List[ModelResult], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)

    # collect all metric keys (union)
    keys = sorted({k for r in results for k in r.metrics.keys()})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_key", "split", "metrics_path"] + keys)
        for r in results:
            row = [r.model_key, r.split_name, str(r.metrics_path)]
            for k in keys:
                row.append(r.metrics.get(k, ""))
            w.writerow(row)


# ---------- OnePass extra chart: skills-only vs typed ----------
def onepass_skills_only_vs_typed_bar(
    results: List[ModelResult],
    metric: str,  # "f1" / "precision" / "recall"
    title: str,
    out_path: Path
) -> None:
    """
    Compare inside OnePass only:
      - skills-only metrics (skill name match ONLY, ignoring type)
      - typed metrics (skill name + type match)

    OnePass keys (from your example):
      skills-only: precision_skills_only / recall_skills_only / f1_skills_only
      typed:       typed_precision_at_k / typed_recall_at_k / typed_f1_at_k
    """
    ensure_dir(out_path.parent)

    onepass = [r for r in results if r.model_key.startswith("onepass_")]
    if not onepass:
        return

    if metric == "f1":
        k_sk = "f1_skills_only"
        k_ty = "typed_f1_at_k"
    elif metric == "precision":
        k_sk = "precision_skills_only"
        k_ty = "typed_precision_at_k"
    elif metric == "recall":
        k_sk = "recall_skills_only"
        k_ty = "typed_recall_at_k"
    else:
        raise ValueError("metric must be one of: f1, precision, recall")

    labels = [r.model_key.replace("onepass_", "") for r in onepass]
    skills_vals = [float(r.metrics.get(k_sk, 0.0)) for r in onepass]
    typed_vals = [float(r.metrics.get(k_ty, 0.0)) for r in onepass]

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(10, 5))
    # IMPORTANT: title/legend make it explicit that skills-only ignores type
    plt.bar([i - width/2 for i in x], skills_vals, width=width,
            label="Skill name only (ignoring type)")
    plt.bar([i + width/2 for i in x], typed_vals, width=width,
            label="Skill + type (typed)")

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def load_model_results(
    split_name: str,
    onepass_runs: Optional[List[str]] = None,
) -> List[ModelResult]:
    """
    Loads:
      - Pairwise:
          trained_models/pairwise/eval_results/**/mean_metrics_val.json
          trained_models/pairwise/test_pairwise/**/mean_metrics_test.json

      - OnePass (YOUR structure):
          trained_models/one_pass/
            eval_onepass/
              seed43_ep3_baseline__VAL/ ... mean_metrics_val.json
              seed43_ep3_cw__VAL/       ... mean_metrics_val.json
            test_onepass/
              seed43_ep3_baseline__TEST/ ... mean_metrics_test.json
              seed43_ep3_cw__TEST/       ... mean_metrics_test.json
    """
    results: List[ModelResult] = []

    # --- Pairwise ---
    pairwise_dir = TRAINED_MODELS_DIR / "pairwise"
    pairwise_base = pairwise_dir / ("test_pairwise" if split_name == "test" else "eval_results")
    pairwise_metrics = find_latest_mean_metrics(pairwise_base, split_name=split_name)

    results.append(
        ModelResult(
            model_key="pairwise",
            split_name=split_name,
            metrics_path=pairwise_metrics,
            metrics={k: float(v) for k, v in read_json(pairwise_metrics).items()},
        )
    )

    # --- OnePass ---
    onepass_dir = TRAINED_MODELS_DIR / "one_pass"
    onepass_base = onepass_dir / ("test_onepass" if split_name == "test" else "eval_onepass")

    if onepass_runs is None or len(onepass_runs) == 0:
        # auto-detect run folders under onepass_base
        if onepass_base.exists():
            onepass_runs = sorted([p.name for p in onepass_base.iterdir() if p.is_dir()])
        else:
            onepass_runs = []

    for run in onepass_runs:
        run_dir = onepass_base / run
        if not run_dir.exists():
            continue

        metrics_path = find_latest_mean_metrics(run_dir, split_name=split_name)
        metrics = {k: float(v) for k, v in read_json(metrics_path).items()}

        results.append(
            ModelResult(
                model_key=f"onepass_{run}",
                split_name=split_name,
                metrics_path=metrics_path,
                metrics=metrics,
            )
        )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument(
        "--onepass_runs",
        nargs="*",
        default=None,
        help=(
            "Optional list of OnePass run folder names.\n"
            "VAL: names under trained_models/one_pass/eval_onepass/\n"
            "TEST: names under trained_models/one_pass/test_onepass/\n"
            "If omitted, auto-detect all folders."
        ),
    )
    args = ap.parse_args()

    if not TRAINED_MODELS_DIR.exists():
        raise FileNotFoundError(f"trained_models folder not found at: {TRAINED_MODELS_DIR}")

    results = load_model_results(split_name=args.split, onepass_runs=args.onepass_runs)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run_dir = OUT_DIR / f"{args.split}_{stamp}"
    ensure_dir(out_run_dir)

    # Save raw summary json
    summary = {
        "repo_root": str(REPO_ROOT),
        "trained_models_dir": str(TRAINED_MODELS_DIR),
        "split": args.split,
        "items": [
            {
                "model_key": r.model_key,
                "metrics_path": str(r.metrics_path),
                "metrics": r.metrics,
            }
            for r in results
        ],
    }
    (out_run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save CSV summary
    save_summary_csv(results, out_run_dir / "summary.csv")

    # 1) Titles WITHOUT "@k" (but values are still computed @K internally)
    #    We also unify Pairwise vs OnePass key names using "__F1__" etc.
    charts = [
        ("__F1__", "F1", "01_f1.png"),
        ("__PREC__", "Precision", "02_precision.png"),
        ("__REC__", "Recall", "03_recall.png"),
        ("type_acc_on_intersection", "Type Accuracy (intersection)", "04_type_acc.png"),
        ("k", "Average K", "05_avg_k.png"),
        ("gt_nonzero", "Average GT nonzero skills", "06_gt_nonzero.png"),
    ]

    for tag, title, fn in charts:
        bar_compare_flexible(results, tag, f"{title} ({args.split})", out_run_dir / fn)

    # 2) Extra OnePass chart(s): "skill name only (ignoring type)" vs "skill+type (typed)"
    onepass_skills_only_vs_typed_bar(
        results,
        metric="f1",
        title=f"OnePass - Skill name only (ignoring type) vs Skill+Type (typed): F1 ({args.split})",
        out_path=out_run_dir / "07_onepass_f1_name_only_vs_typed.png",
    )
    onepass_skills_only_vs_typed_bar(
        results,
        metric="precision",
        title=f"OnePass - Skill name only (ignoring type) vs Skill+Type (typed): Precision ({args.split})",
        out_path=out_run_dir / "08_onepass_precision_name_only_vs_typed.png",
    )
    onepass_skills_only_vs_typed_bar(
        results,
        metric="recall",
        title=f"OnePass - Skill name only (ignoring type) vs Skill+Type (typed): Recall ({args.split})",
        out_path=out_run_dir / "09_onepass_recall_name_only_vs_typed.png",
    )

    print("[OK] Repo root:", REPO_ROOT)
    print("[OK] Using trained_models:", TRAINED_MODELS_DIR)
    print("[OK] Saved outputs to:", out_run_dir)
    print("\nLoaded models:")
    for r in results:
        print(f" - {r.model_key}: {r.metrics_path}")


if __name__ == "__main__":
    main()
