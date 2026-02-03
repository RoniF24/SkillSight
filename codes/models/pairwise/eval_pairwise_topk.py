"""codes/models/pairwise/eval_pairwise_topk.py

JD-level top-k evaluation for the Pairwise model.

High-level flow for each job description:
- Score every skill in `skills_used.txt` with the trained pairwise classifier.
    The model outputs probabilities for {NONE, IMPLICIT, EXPLICIT}.
- Convert to a single per-skill score (`nonzero_score = 1 - p(NONE)`) and a type label
    (IMPLICIT vs EXPLICIT from comparing p1 vs p2).
- Select k skills (k in [3..6]) with an optional gap rule and an optional prefilter.
- Evaluate predictions against the JSONL ground truth.

Important note about correctness:
- JD-level metrics here are *typed*: a prediction is counted as correct only if both
    the skill name matches and the type matches (0.5 for implicit vs 1.0 for explicit).

Outputs:
- `predictions_<split>.jsonl` with per-item predictions and metrics.
- `mean_metrics_<split>.json` with the dataset mean of the JD-level metrics.
- `run_config.json` capturing all args and paths used.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Repo root is three levels above this file (SkillSight-main/).
REPO_ROOT = Path(__file__).resolve().parents[3]

# Where pairwise models are saved (relative to repo root).
PAIRWISE_MODEL_BASE = REPO_ROOT / "trained pairwise"


# ---------------- IO ----------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- Data access helpers ----------------
def get_id(item: Dict[str, Any]) -> str:
    for k in ["id", "uid", "hash", "example_id"]:
        if k in item:
            return str(item[k])
    return str(abs(hash(item.get("job_description", ""))))  # fallback


def get_text(item: Dict[str, Any]) -> str:
    for k in ["job_description", "text", "jd"]:
        if k in item:
            return str(item[k])
    raise KeyError("Could not find job description text field (job_description/text/jd).")


def get_gt_skills(item: Dict[str, Any]) -> Dict[str, float]:
    skills = item.get("skills", {})
    if not isinstance(skills, dict):
        return {}
    # keep only >0
    return {s: float(v) for s, v in skills.items() if float(v) > 0.0}


# ---------------- Model scoring ----------------
@torch.no_grad()
def predict_scores_for_item(
    model,
    tokenizer,
    text: str,
    skills_list: List[str],
    batch_size: int,
    device: str,
) -> List[Tuple[str, float, float, float, float, int]]:
    """
    For each skill returns:
    (skill, nonzero_score, p0, p1, p2, type_label)
    type_label: 1=IMPLICIT, 2=EXPLICIT (p1 vs p2)
    """
    out: List[Tuple[str, float, float, float, float, int]] = []
    model.eval()

    for i in range(0, len(skills_list), batch_size):
        batch_skills = skills_list[i : i + batch_size]
        enc = tokenizer(
            batch_skills,
            [text] * len(batch_skills),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits  # [B,3]
        probs = torch.softmax(logits, dim=-1)
        p0 = probs[:, 0]
        p1 = probs[:, 1]
        p2 = probs[:, 2]

        nonzero = 1.0 - p0
        type_label = torch.where(
            p2 > p1,
            torch.tensor(2, device=device),
            torch.tensor(1, device=device),
        )

        for s, nz, a0, a1, a2, tl in zip(batch_skills, nonzero, p0, p1, p2, type_label):
            out.append((s, float(nz.item()), float(a0.item()), float(a1.item()), float(a2.item()), int(tl.item())))

    return out


# ---------------- Top-K (3..6) selection ----------------
def pick_topk_3_to_6(
    scored: List[Tuple[str, float, float, float, float, int]],
    k_max: int = 6,
    k_min: int = 3,
    gap_ratio: float = 0.25,
    use_gap_rule: bool = True,
    prefilter_threshold: Optional[float] = None,
    keep6_min_score: Optional[float] = None,
) -> List[Tuple[str, float, int]]:
    """
    scored: (skill, nonzero_score, p0,p1,p2,type_label)
    returns: (skill, nonzero_score, type_label) size 3..6
    """
    if prefilter_threshold is not None:
        scored = [x for x in scored if x[1] >= prefilter_threshold]

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    if not scored_sorted:
        return []

    take = min(k_max, len(scored_sorted))

    if use_gap_rule and take > k_min:
        best_cut = None
        for j in range(k_min, take):
            prev = scored_sorted[j - 1][1]
            curr = scored_sorted[j][1]
            rel_gap = (prev - curr) / max(prev, 1e-9)
            if rel_gap >= gap_ratio:
                best_cut = j
                break
        if best_cut is not None:
            take = best_cut

    if keep6_min_score is not None and take >= 6 and len(scored_sorted) >= 6:
        sixth_score = scored_sorted[5][1]
        if sixth_score < keep6_min_score:
            take = 5

    take = max(k_min, take) if len(scored_sorted) >= k_min else len(scored_sorted)
    picked = scored_sorted[:take]
    return [(s, nz, tl) for (s, nz, _p0, _p1, _p2, tl) in picked]


# ---------------- Metrics (JD-level) ----------------
def jd_level_metrics(
    pred: List[Tuple[str, float, int]],
    gt: Dict[str, float],
) -> Dict[str, float]:
    """
    JD-level metrics with type awareness ("typed" correctness).

    A prediction counts as correct only if:
    - skill name matches, and
    - type matches exactly (0.5 vs 1.0).
    """

    # pred: list[(skill, nz, type_label)] where type_label 1->0.5, 2->1.0
    pred_map: Dict[str, float] = {s: (0.5 if tl == 1 else 1.0) for (s, _, tl) in pred}
    gt_map: Dict[str, float] = {s: float(v) for s, v in gt.items()}

    pred_set = set(pred_map.keys())
    gt_set = set(gt_map.keys())

    k = len(pred_set)
    gt_n = len(gt_set)

    # "typed correct": same skill AND same value
    typed_correct = {s for s in (pred_set & gt_set) if float(pred_map[s]) == float(gt_map[s])}
    typed_correct_n = len(typed_correct)

    # typed precision/recall/f1
    precision = (typed_correct_n / k) if k > 0 else 0.0
    recall = (typed_correct_n / gt_n) if gt_n > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # for interpretation:
    # how many skills matched by name (ignoring type)
    name_inter = pred_set & gt_set
    name_inter_n = len(name_inter)

    # type accuracy on name-intersection (old meaning):
    type_acc_on_intersection = (typed_correct_n / name_inter_n) if name_inter_n > 0 else 0.0
    type_support_on_intersection = float(name_inter_n)

    # optional: exact match (skill+type) of the whole dictionary
    exact_match_with_type = 1.0 if pred_map == gt_map else 0.0

    return {
        # NOW these are typed metrics (skill+type)
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),

        # still useful (interpretation)
        "typed_correct_count": float(typed_correct_n),
        "k": float(k),
        "gt_nonzero": float(gt_n),

        # compatibility / debug
        "type_acc_on_intersection": float(type_acc_on_intersection),
        "type_support_on_intersection": float(type_support_on_intersection),
        "name_intersection_count": float(name_inter_n),
        "exact_match_with_type": float(exact_match_with_type),
    }


# ---------------- Defaults (stable across runs) ----------------
def find_latest_model_final(models_base_dir: Path) -> Path:
    candidates = list(models_base_dir.glob("**/final"))
    candidates = [p for p in candidates if p.is_dir() and (p / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"Could not find any trained model 'final' under: {models_base_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_run_name(
    k_min: int,
    k_max: int,
    use_gap_rule: bool,
    gap_ratio: float,
    prefilter_threshold: Optional[float],
) -> str:
    parts = [f"topk_{k_min}_{k_max}"]
    if use_gap_rule:
        parts.append(f"gap_{gap_ratio:.2f}".replace(".", "p"))
    else:
        parts.append("nogap")
    if prefilter_threshold is not None:
        parts.append(f"pref_{prefilter_threshold:.2f}".replace(".", "p"))
    return "__".join(parts)


def main():
    trained_models_dir = REPO_ROOT / "trained_models"

    default_model_dir = find_latest_model_final(PAIRWISE_MODEL_BASE)
    default_data_jsonl = REPO_ROOT / "data" / "splits_v1" / "val.jsonl"
    default_skills_file = REPO_ROOT / "data" / "splits_v1" / "skills_used.txt"

    default_k_min = 3
    default_k_max = 6
    default_use_gap_rule = True
    default_gap_ratio = 0.25
    default_prefilter_threshold: Optional[float] = None
    default_keep6_min_score: Optional[float] = 0.99

    default_run_name = build_run_name(
        default_k_min, default_k_max, default_use_gap_rule, default_gap_ratio, default_prefilter_threshold
    )
    if default_keep6_min_score is not None:
        default_run_name = default_run_name + f"__keep6_{str(default_keep6_min_score).replace('.', 'p')}"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(default_model_dir))
    ap.add_argument("--data_jsonl", type=str, default=str(default_data_jsonl))
    ap.add_argument("--skills_file", type=str, default=str(default_skills_file))

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--k_min", type=int, default=default_k_min)
    ap.add_argument("--k_max", type=int, default=default_k_max)

    ap.add_argument("--use_gap_rule", action="store_true", default=default_use_gap_rule)
    ap.add_argument("--gap_ratio", type=float, default=default_gap_ratio)

    ap.add_argument("--prefilter_threshold", type=float, default=default_prefilter_threshold)

    ap.add_argument(
        "--keep6_min_score",
        type=float,
        default=default_keep6_min_score,
        help="Return 6 skills only if 6th nonzero_score >= this value; else return 5 (still min 3).",
    )

    ap.add_argument("--split_name", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--run_name", type=str, default=default_run_name)

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data_jsonl)
    skills_path = Path(args.skills_file)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_jsonl not found: {data_path}")
    if not skills_path.exists():
        raise FileNotFoundError(f"skills_file not found: {skills_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = (trained_models_dir / "test_pairwise") if args.split_name == "test" else (trained_models_dir / "eval_results")
    out_dir = base_dir / args.run_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pred_full_jsonl = out_dir / f"predictions_{args.split_name}.jsonl"
    out_mean_metrics = out_dir / f"mean_metrics_{args.split_name}.json"
    out_run_config = out_dir / "run_config.json"
    out_pred_clean_jsonl = out_dir / "predictions_test_clean.jsonl"

    skills_list = [ln.strip() for ln in skills_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(args.device)

    items = read_jsonl(data_path)
    n = len(items)

    preds_rows: List[Dict[str, Any]] = []

    # Aggregation keys match `jd_level_metrics()` output.
    agg = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_at_k": 0.0,

        "typed_correct_count": 0.0,
        "k": 0.0,
        "gt_nonzero": 0.0,

        "type_acc_on_intersection": 0.0,
        "type_support_on_intersection": 0.0,
        "name_intersection_count": 0.0,
        "exact_match_with_type": 0.0,
    }

    for idx, item in enumerate(items, start=1):
        ex_id = get_id(item)
        text = get_text(item)
        gt = get_gt_skills(item)

        scored = predict_scores_for_item(
            model=model,
            tokenizer=tokenizer,
            text=text,
            skills_list=skills_list,
            batch_size=args.batch_size,
            device=args.device,
        )

        picked = pick_topk_3_to_6(
            scored,
            k_max=args.k_max,
            k_min=args.k_min,
            gap_ratio=args.gap_ratio,
            use_gap_rule=args.use_gap_rule,
            prefilter_threshold=args.prefilter_threshold,
            keep6_min_score=args.keep6_min_score,
        )

        if len(picked) < args.k_min:
            picked = pick_topk_3_to_6(
                scored,
                k_max=args.k_max,
                k_min=args.k_min,
                gap_ratio=args.gap_ratio,
                use_gap_rule=args.use_gap_rule,
                prefilter_threshold=None,
                keep6_min_score=args.keep6_min_score,
            )

        pred_skills_dict: Dict[str, float] = {}
        pred_list: List[Dict[str, Any]] = []
        for s, nz, tl in picked:
            val = 0.5 if tl == 1 else 1.0
            pred_skills_dict[s] = val
            pred_list.append({"skill": s, "score": val, "nonzero_score": nz})

        m = jd_level_metrics(picked, gt)
        for kk in agg:
            agg[kk] += float(m.get(kk, 0.0))

        preds_rows.append(
            {
                "id": ex_id,
                "job_description": text,
                "predicted": pred_list,
                "predicted_skills": pred_skills_dict,
                "gt_skills": gt,
                "metrics": m,
            }
        )

        if idx % 25 == 0 or idx == n:
            print(f"[{idx}/{n}] processed...")

    mean = {kk: (agg[kk] / n if n > 0 else 0.0) for kk in agg}

    print("\n=== JD-level MEAN metrics (PAIRWISE, TYPED) ===")
    for kk, v in mean.items():
        print(f"{kk}: {v:.4f}")

    write_jsonl(out_pred_full_jsonl, preds_rows)
    write_json(out_mean_metrics, mean)

    if args.split_name == "test":
        clean_rows = [{"id": r["id"], "job_description": r["job_description"], "predicted_skills": r["predicted_skills"]} for r in preds_rows]
        write_jsonl(out_pred_clean_jsonl, clean_rows)

    run_config = {
        "repo_root": str(REPO_ROOT),
        "model_dir": str(model_dir),
        "data_jsonl": str(data_path),
        "skills_file": str(skills_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "use_gap_rule": bool(args.use_gap_rule),
        "gap_ratio": args.gap_ratio,
        "prefilter_threshold": args.prefilter_threshold,
        "keep6_min_score": args.keep6_min_score,
        "split_name": args.split_name,
        "out_dir": str(out_dir),
        "outputs": {
            "predictions_full_jsonl": str(out_pred_full_jsonl),
            "predictions_test_clean_jsonl": str(out_pred_clean_jsonl) if args.split_name == "test" else None,
            "mean_metrics_json": str(out_mean_metrics),
        },
        "num_items": n,
        "metrics_mode": "TYPED (skill+type must match)",
    }
    write_json(out_run_config, run_config)

    print(f"\nSaved FULL predictions to: {out_pred_full_jsonl}")
    if args.split_name == "test":
        print(f"Saved CLEAN test predictions to: {out_pred_clean_jsonl}")
    print(f"Saved mean metrics to: {out_mean_metrics}")
    print(f"Saved run config to: {out_run_config}")
    print(f"Results folder: {out_dir}")


if __name__ == "__main__":
    main()
