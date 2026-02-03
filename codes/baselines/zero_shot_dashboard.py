from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required. Install: pip install matplotlib") from e


# -------------------- IO helpers --------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def infer_dataset_name(run_dir: Path) -> str:
    """
    מנסה לנחש שם דאטהסט מתוך שם התיקייה.
    אצלך זה בד"כ run__zero_shot_<model>_<timestamp>
    """
    name = run_dir.name
    if name.startswith("run__"):
        return name.replace("run__", "")
    return name


# -------------------- Metrics parsing --------------------

def _norm_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    m = re.sub(r"\s+", "_", m)
    return m


def _classify_mode(mode: str) -> Optional[str]:
    """
    מיפוי עמיד לרשומת 'mode' מתוך overall_metrics.csv.

    חשוב: any חייב להיבדק לפני explicit וכו'.
    """
    m = _norm_mode(mode)

    # Any (>=0.5 / 0.5_or_1.0 / any_evidence)
    if (
        "any_evidence" in m
        or "0.5_or_1.0" in m
        or ">=0.5" in m
        or re.search(r"(^|_)any($|_)", m)
    ):
        return "any"

    # Implicit only (==0.5)
    if (
        "implicit_only" in m
        or "implicit" in m
        or "0.5_only" in m
        or "==0.5" in m
    ):
        return "implicit"

    # Explicit only (>=1.0 / ==1.0)
    if (
        "explicit_only" in m
        or "explicit" in m
        or ">=1.0" in m
        or "==1.0" in m
        or "1.0_only" in m
    ):
        return "explicit"

    return None


def _mode_priority(mode: str, key: str) -> int:
    """
    אם יש כמה שורות שממופות לאותו key, נשמור את הכי ספציפית.
    מספר גדול = יותר ספציפי.
    """
    m = _norm_mode(mode)

    if key == "any":
        if "any_evidence" in m or "0.5_or_1.0" in m or ">=0.5" in m:
            return 3
        if re.search(r"(^|_)any($|_)", m):
            return 2
        return 1

    if key == "implicit":
        if "implicit_only" in m or "==0.5" in m or "0.5_only" in m:
            return 3
        if "implicit" in m:
            return 2
        return 1

    if key == "explicit":
        if "explicit_only" in m or ">=1.0" in m or "==1.0" in m or "1.0_only" in m:
            return 3
        if "explicit" in m:
            return 2
        return 1

    return 0


def load_overall_metrics(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    קורא overall_metrics.csv ומחזיר dict עם:
      - 'explicit' / 'implicit' / 'any'
      לכל אחד: precision, recall, f1, tp, fp, fn
    """
    rows = read_csv_rows(run_dir / "overall_metrics.csv")
    out: Dict[str, Dict[str, float]] = {}
    best_pri: Dict[str, int] = {}

    for r in rows:
        mode_raw = (r.get("mode") or "")
        key = _classify_mode(mode_raw)
        if not key:
            continue

        pri = _mode_priority(mode_raw, key)
        if key in best_pri and pri < best_pri[key]:
            continue

        out[key] = {
            "precision": parse_float(r.get("precision"), 0.0),
            "recall": parse_float(r.get("recall"), 0.0),
            "f1": parse_float(r.get("f1"), 0.0),
            "tp": parse_float(r.get("tp"), 0.0),
            "fp": parse_float(r.get("fp"), 0.0),
            "fn": parse_float(r.get("fn"), 0.0),
        }
        best_pri[key] = pri

    return out


def load_per_skill_f1(run_dir: Path, mode: str) -> List[Tuple[str, float]]:
    """
    מחזיר [(skill, f1), ...] עבור any / implicit / explicit.
    """
    mode = mode.lower().strip()
    if mode == "any":
        path = run_dir / "per_skill__any_evidence__f1.csv"
    elif mode == "implicit":
        path = run_dir / "per_skill__implicit_only__f1.csv"
    elif mode == "explicit":
        path = run_dir / "per_skill__explicit_only__f1.csv"
    else:
        raise ValueError("mode must be: any/implicit/explicit")

    rows = read_csv_rows(path)
    items: List[Tuple[str, float]] = []
    for r in rows:
        s = (r.get("skill") or "").strip()
        if not s:
            continue
        items.append((s, parse_float(r.get("f1"), 0.0)))
    return items


def count_errors(run_dir: Path, mode: str) -> Tuple[int, int]:
    """
    סופר כמה FP וכמה FN יש בקבצי הטעויות.
    """
    mode = mode.lower().strip()
    if mode == "any":
        path = run_dir / "errors__any_evidence__fp_fn.csv"
    elif mode == "implicit":
        path = run_dir / "errors__implicit_only__fp_fn.csv"
    elif mode == "explicit":
        path = run_dir / "errors__explicit_only__fp_fn.csv"
    else:
        raise ValueError("mode must be: any/implicit/explicit")

    rows = read_csv_rows(path)
    fp = sum(1 for r in rows if (r.get("type") or "").upper() == "FP")
    fn = sum(1 for r in rows if (r.get("type") or "").upper() == "FN")
    return fp, fn


def top_bottom(items: List[Tuple[str, float]], k: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if not items:
        return [], []
    items_desc = sorted(items, key=lambda x: x[1], reverse=True)
    topk = items_desc[:k]
    bottomk = sorted(items_desc, key=lambda x: x[1])[:k]
    return topk, bottomk


# -------------------- Plot helpers --------------------

def _plot_barh_with_zero_visible(ax, items: List[Tuple[str, float]], title: str, missing_msg: str) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold")
    if not items:
        ax.axis("off")
        ax.text(0.5, 0.5, missing_msg, ha="center", va="center")
        return

    skills = [s for s, _ in items]
    vals = [v for _, v in items]

    # גרימה לערכי 0 להיות נראים – נשים epsilon קטן
    eps = 0.01
    plot_vals = [v if v > 0 else eps for v in vals]

    ax.barh(list(reversed(skills)), list(reversed(plot_vals)))
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("F1")

    # כותבים את הערכים האמיתיים (כולל 0.00)
    for i, (s, v) in enumerate(reversed(items)):
        x = (v if v > 0 else eps)
        ax.text(min(x + 0.02, 1.02), i, f"{v:.2f}", va="center", fontsize=10)


# -------------------- Main dashboard --------------------

def make_zero_shot_dashboard(run_dir: Path, out_name: str) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    dataset = infer_dataset_name(run_dir)
    samples = count_jsonl_lines(run_dir / "preds.jsonl")

    overall = load_overall_metrics(run_dir)
    exp = overall.get("explicit")
    imp = overall.get("implicit")
    any_m = overall.get("any")

    per_any = load_per_skill_f1(run_dir, "any")
    per_imp = load_per_skill_f1(run_dir, "implicit")

    top_any, bot_any = top_bottom(per_any, 5)
    top_imp, bot_imp = top_bottom(per_imp, 5)

    fp_any, fn_any = count_errors(run_dir, "any")
    fp_imp, fn_imp = count_errors(run_dir, "implicit")

    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.05, 1.0])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_micro = fig.add_subplot(gs[0, 1])
    ax_any_top = fig.add_subplot(gs[1, 0])
    ax_imp_top = fig.add_subplot(gs[1, 1])


    fig.suptitle(f"Dashboard — zero_shot | run: {dataset}", fontsize=22, fontweight="bold", y=0.985)

    ax_text.axis("off")

    def fmt_block(name: str, m: Optional[Dict[str, float]]) -> str:
        if not m:
            return f"- {name}: (missing)"
        return (
            f"- {name}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"(TP/FP/FN={int(m['tp'])}/{int(m['fp'])}/{int(m['fn'])})"
        )

    lines = [
        "Summary",
        f"- Run dir: {run_dir.name}",
        f"- Samples evaluated: {samples}",
        "",
        "Micro Metrics",
        fmt_block("Explicit (>=1.0)", exp),
        fmt_block("Implicit (==0.5)", imp),
        fmt_block("Any (>=0.5)", any_m),
        "",
        f"Errors Any: FP={fp_any} | FN={fn_any}",
        f"Errors Implicit: FP={fp_imp} | FN={fn_imp}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=12)

    # Micro plot
    ax_micro.set_title("Micro metrics (Explicit / Implicit / Any)", fontsize=16, fontweight="bold")
    labels = ["PRECISION", "RECALL", "F1"]
    x = list(range(len(labels)))

    series: List[List[float]] = []
    legends: List[str] = []

    if exp:
        series.append([exp["precision"], exp["recall"], exp["f1"]])
        legends.append("Explicit (>=1.0)")
    if imp:
        series.append([imp["precision"], imp["recall"], imp["f1"]])
        legends.append("Implicit (==0.5)")
    if any_m:
        series.append([any_m["precision"], any_m["recall"], any_m["f1"]])
        legends.append("Any (>=0.5)")

    if not series:
        ax_micro.text(0.5, 0.5, "overall_metrics.csv missing/empty", ha="center", va="center")
        ax_micro.set_xticks([])
        ax_micro.set_yticks([])
    else:
        n = len(series)
        total_width = 0.8
        bar_w = total_width / n
        center_offset = (n - 1) / 2.0

        for idx, vals in enumerate(series):
            offset = (idx - center_offset) * bar_w
            xpos = [i + offset for i in x]
            ax_micro.bar(xpos, vals, width=bar_w)

        ax_micro.set_ylim(0.0, 1.05)
        ax_micro.set_xticks(x)
        ax_micro.set_xticklabels(labels)
        ax_micro.legend(legends, loc="upper right")

    # Any top/bottom
    _plot_barh_with_zero_visible(
        ax_any_top,
        top_any,
        "Top 5 skills by F1 (Any >=0.5)",
        "Missing per_skill__any_evidence__f1.csv",
    )

    # Implicit top/bottom
    _plot_barh_with_zero_visible(
        ax_imp_top,
        top_imp,
        "Top 5 skills by F1 (Implicit ==0.5)",
        "Missing per_skill__implicit_only__f1.csv",
    )


    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Path to run__ directory (with overall_metrics.csv, per_skill__*.csv, errors__*.csv, preds.jsonl)",
    )
    ap.add_argument(
        "--out_name",
        default="dashboard__zero_shot.png",
        help="Output PNG file name (inside run_dir)",
    )
    args = ap.parse_args()

    out = make_zero_shot_dashboard(Path(args.run_dir), args.out_name)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
