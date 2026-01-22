from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set

# נגדיר שורש ריפו כדי להגיע לדאטהסט המקורי (gold)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2] if len(THIS_FILE.parents) >= 3 else Path.cwd()
DATA_PATH = REPO_ROOT / "data" / "synthetic_dataset.jsonl"


def load_gold_dataset(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON in gold dataset on line {line_no}: {e}") from e
            if "job_description" not in obj or "skills" not in obj:
                raise ValueError(f"Missing keys in gold dataset on line {line_no}: {obj.keys()}")
            examples.append(obj)
    return examples


def get_gold_labels(skills: Dict[str, float], mode: str) -> List[str]:
    """
    חייב להיות עקבי עם get_gold_labels ב-pure_zero_shot.py

    mode:
      - 'explicit'  -> weight >= 0.99
      - 'implicit'  -> 0.49 <= weight < 0.99
      - 'any'       -> weight >= 0.49
    """
    if mode == "explicit":
        return [s for s, w in skills.items() if w >= 0.99]
    elif mode == "implicit":
        return [s for s, w in skills.items() if 0.49 <= w < 0.99]
    elif mode == "any":
        return [s for s, w in skills.items() if w >= 0.49]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_pred_labels(skills: Dict[str, float], mode: str) -> List[str]:
    """
    בוחר סקילים לפי התחזיות של המודל:
      - 'explicit'  -> predicted >= 0.99
      - 'implicit'  -> 0.49 <= pred < 0.99
      - 'any'       -> pred >= 0.49
    """
    if mode == "explicit":
        return [s for s, w in skills.items() if w >= 0.99]
    elif mode == "implicit":
        return [s for s, w in skills.items() if 0.49 <= w < 0.99]
    elif mode == "any":
        return [s for s, w in skills.items() if w >= 0.49]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_stats(preds_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # נעתיק את ה-JSONL לתוך run_dir בשם preds.jsonl (הדשבורד מצפה לזה)
    preds_copy = run_dir / "preds.jsonl"
    preds_copy.write_text(preds_path.read_text(encoding="utf-8"), encoding="utf-8")

    # טוענים את הדאטהסט המקורי כ-gold
    gold_examples = load_gold_dataset(DATA_PATH)

    modes = ["explicit", "implicit", "any"]

    # tp/fp/fn כללי
    overall_counts: Dict[str, Dict[str, int]] = {
        m: {"tp": 0, "fp": 0, "fn": 0} for m in modes
    }

    # tp/fp/fn לכל סקיל
    per_skill_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        m: {} for m in modes
    }

    # רשימות שגיאות FP/FN
    error_rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    with preds_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            ex_pred = json.loads(line)

            # התאמה לדוגמה המתאימה בדאטהסט (לפי אינדקס)
            if idx > len(gold_examples):
                break
            ex_gold = gold_examples[idx - 1]

            job_description: str = ex_pred.get("job_description", "") or ""
            gold_skills: Dict[str, float] = ex_gold.get("skills", {}) or {}
            pred_skills: Dict[str, float] = ex_pred.get("skills", {}) or {}

            for mode in modes:
                gold_labels = get_gold_labels(gold_skills, mode)
                pred_labels = get_pred_labels(pred_skills, mode)

                gold_set: Set[str] = set(gold_labels)
                pred_set: Set[str] = set(pred_labels)

                # FNs: ב-gold ולא ב-pred
                for skill in gold_set:
                    if skill not in pred_set:
                        overall_counts[mode]["fn"] += 1
                        per_skill_counts[mode].setdefault(skill, {"tp": 0, "fp": 0, "fn": 0})
                        per_skill_counts[mode][skill]["fn"] += 1
                        error_rows[mode].append(
                            {
                                "type": "FN",
                                "skill": skill,
                                "example_idx": idx,
                                "job_description": job_description,
                            }
                        )

                # FPs + TPs
                for skill in pred_set:
                    per_skill_counts[mode].setdefault(skill, {"tp": 0, "fp": 0, "fn": 0})
                    if skill in gold_set:
                        overall_counts[mode]["tp"] += 1
                        per_skill_counts[mode][skill]["tp"] += 1
                    else:
                        overall_counts[mode]["fp"] += 1
                        per_skill_counts[mode][skill]["fp"] += 1
                        error_rows[mode].append(
                            {
                                "type": "FP",
                                "skill": skill,
                                "example_idx": idx,
                                "job_description": job_description,
                            }
                        )

    # ---------- overall_metrics.csv ----------

    overall_csv = run_dir / "overall_metrics.csv"
    with overall_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mode", "precision", "recall", "f1", "tp", "fp", "fn"]
        )
        writer.writeheader()

        for mode in modes:
            c = overall_counts[mode]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

            if mode == "any":
                mode_str = "any_evidence (>=0.5 or 1.0)"
            elif mode == "implicit":
                mode_str = "implicit_only (==0.5)"
            else:
                mode_str = "explicit_only (>=1.0)"

            writer.writerow(
                {
                    "mode": mode_str,
                    "precision": f"{prec:.6f}",
                    "recall": f"{rec:.6f}",
                    "f1": f"{f1:.6f}",
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

    # ---------- per_skill__*_f1.csv ----------

    per_skill_files = {
        "any": run_dir / "per_skill__any_evidence__f1.csv",
        "implicit": run_dir / "per_skill__implicit_only__f1.csv",
        "explicit": run_dir / "per_skill__explicit_only__f1.csv",
    }

    for mode in modes:
        path = per_skill_files[mode]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["skill", "precision", "recall", "f1", "tp", "fp", "fn"],
            )
            writer.writeheader()

            for skill, c in sorted(per_skill_counts[mode].items()):
                tp, fp, fn = c["tp"], c["fp"], c["fn"]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

                writer.writerow(
                    {
                        "skill": skill,
                        "precision": f"{prec:.6f}",
                        "recall": f"{rec:.6f}",
                        "f1": f"{f1:.6f}",
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

    # ---------- errors__*_fp_fn.csv ----------

    error_files = {
        "any": run_dir / "errors__any_evidence__fp_fn.csv",
        "implicit": run_dir / "errors__implicit_only__fp_fn.csv",
        "explicit": run_dir / "errors__explicit_only__fp_fn.csv",
    }

    for mode, path in error_files.items():
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["type", "skill", "example_idx", "job_description"],
            )
            writer.writeheader()
            for row in error_rows[mode]:
                writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze zero-shot predictions.jsonl and create CSV metrics + errors."
    )
    ap.add_argument("--preds", required=True, help="Path to *_predictions.jsonl from pure_zero_shot.py")
    ap.add_argument(
        "--run-dir",
        help="Output directory for CSVs + preds.jsonl. "
             "Default: <preds_dir>/run__<preds_stem>",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(f"preds file not found: {preds_path}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        base = preds_path.stem.replace("_predictions", "")
        run_dir = preds_path.parent / f"run__{base}"

    print(f"[INFO] Using gold dataset from: {DATA_PATH}")
    build_stats(preds_path, run_dir)
    print(f"[OK] wrote analysis CSVs to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
