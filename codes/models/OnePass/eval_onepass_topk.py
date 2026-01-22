from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---------------- Repo root finder ----------------
def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start.parent] + list(start.parents):
        if (p / "codes").exists() and (p / "data").exists():
            return p
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()

# where OnePass models are saved (outside codes)
ONEPASS_MODEL_BASE = REPO_ROOT / "trained onepass"

# --- local imports ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_onepass import read_jsonl  # noqa: E402


# ---------------- IO ----------------
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    raise KeyError("Could not find text field (job_description/text/jd).")


def get_gt_skills(item: Dict[str, Any]) -> Dict[str, float]:
    skills = item.get("skills", {})
    if not isinstance(skills, dict):
        return {}
    return {s: float(v) for s, v in skills.items() if float(v) > 0.0}


# ---------------- OnePass model (must match training) ----------------
class OnePassModel(nn.Module):
    """
    Text -> logits [B, S, 3] where classes: 0 NONE, 1 IMPLICIT, 2 EXPLICIT
    """

    def __init__(self, model_name: str, num_skills: int, dropout: float = 0.1):
        super().__init__()
        self.num_skills = num_skills
        self.num_classes = 3

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        hidden = self.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_skills * self.num_classes)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B,H]
        logits = self.head(self.drop(cls)).view(-1, self.num_skills, self.num_classes)  # [B,S,3]
        return logits


def infer_model_name_from_final(final_dir: Path) -> str:
    meta_path = final_dir / "onepass_meta.json"
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            name = str(meta.get("base_model_name", "")).strip()
            if name:
                return name
        except Exception:
            pass

    cfg = AutoConfig.from_pretrained(str(final_dir))
    name = getattr(cfg, "_name_or_path", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "roberta-base"


def load_state_dict_any(final_dir: Path) -> Dict[str, torch.Tensor]:
    safe_path = final_dir / "model.safetensors"
    bin_path = final_dir / "pytorch_model.bin"

    if safe_path.exists():
        try:
            from safetensors.torch import load_file as safe_load  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Found model.safetensors but safetensors isn't installed. Run: pip install safetensors"
            ) from e
        return safe_load(str(safe_path))

    if bin_path.exists():
        obj = torch.load(str(bin_path), map_location="cpu")
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Unexpected content in {bin_path} (expected state_dict dict).")

    raise FileNotFoundError(f"Missing weights in {final_dir} (need model.safetensors or pytorch_model.bin).")


def load_onepass_from_final(final_dir: Path, num_skills: int, device: str) -> Tuple[nn.Module, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(str(final_dir), use_fast=True)

    model_name = infer_model_name_from_final(final_dir)
    model = OnePassModel(model_name=model_name, num_skills=num_skills)

    state = load_state_dict_any(final_dir)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model, tokenizer, model_name


def find_latest_model_final(models_base_dir: Path) -> Path:
    candidates = list(models_base_dir.glob("**/final"))

    def ok(p: Path) -> bool:
        if not p.is_dir():
            return False
        has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        has_cfg = (p / "onepass_meta.json").exists() or (p / "config.json").exists()
        return has_weights and has_cfg

    candidates = [p for p in candidates if ok(p)]
    if not candidates:
        raise FileNotFoundError(f"Could not find any trained OnePass model 'final' under: {models_base_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ---------------- Top-K (3..6) selection ----------------
def pick_topk_3_to_6_onepass(
    scored: List[Tuple[str, float, float, float, float, int]],
    k_max: int = 6,
    k_min: int = 3,
    gap_ratio: float = 0.25,
    use_gap_rule: bool = True,
    prefilter_threshold: Optional[float] = None,
    keep6_min_score: Optional[float] = 0.99,
) -> List[Tuple[str, float, int]]:
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
        if scored_sorted[5][1] < keep6_min_score:
            take = 5

    take = max(k_min, take) if len(scored_sorted) >= k_min else len(scored_sorted)
    picked = scored_sorted[:take]
    return [(s, nz, tl) for (s, nz, _p0, _p1, _p2, tl) in picked]


# ---------------- Metrics (JD-level) ----------------
def jd_level_metrics(pred: List[Tuple[str, float, int]], gt: Dict[str, float]) -> Dict[str, float]:
    """
    ✅ "בדיקה לפי סיווג" (Typed):
    הצלחה = skill נכון + label נכון (0.5/1.0).

    מחזיר גם:
    - skills-only (שם בלבד) כדי שתוכלי להשוות.
    """

    pred_set = {s for s, _, _ in pred}
    gt_set = set(gt.keys())
    inter_names = pred_set & gt_set

    k = len(pred_set)
    gt_n = len(gt_set)

    # ---------------- skills-only (name only) ----------------
    precision_skills_only = (len(inter_names) / k) if k > 0 else 0.0
    recall_skills_only = (len(inter_names) / gt_n) if gt_n > 0 else 0.0
    f1_skills_only = (
        (2 * precision_skills_only * recall_skills_only / (precision_skills_only + recall_skills_only))
        if (precision_skills_only + recall_skills_only) > 0
        else 0.0
    )

    # build predicted map: skill -> {0.5, 1.0}
    pred_map: Dict[str, float] = {s: (0.5 if tl == 1 else 1.0) for (s, _, tl) in pred}

    # ---------------- typed correctness ----------------
    typed_correct = 0
    for s in inter_names:
        if float(pred_map.get(s, 0.0)) == float(gt[s]):
            typed_correct += 1

    typed_correct_count = float(typed_correct)

    typed_precision_at_k = (typed_correct_count / float(k)) if k > 0 else 0.0
    typed_recall_at_k = (typed_correct_count / float(gt_n)) if gt_n > 0 else 0.0
    typed_f1_at_k = (
        (2 * typed_precision_at_k * typed_recall_at_k / (typed_precision_at_k + typed_recall_at_k))
        if (typed_precision_at_k + typed_recall_at_k) > 0
        else 0.0
    )

    # keep this (useful diagnostic): type accuracy among intersection
    type_acc_on_intersection = (typed_correct_count / float(len(inter_names))) if len(inter_names) > 0 else 0.0
    type_support_on_intersection = float(len(inter_names))

    exact_match_skills_only = 1.0 if pred_set == gt_set else 0.0
    exact_match_with_type = 1.0 if pred_map == {s: float(v) for s, v in gt.items()} else 0.0

    return {
        # ✅ your "real" typed metrics
        "typed_precision_at_k": float(typed_precision_at_k),
        "typed_recall_at_k": float(typed_recall_at_k),
        "typed_f1_at_k": float(typed_f1_at_k),
        "typed_correct_count": float(typed_correct_count),

        # diagnostics
        "type_acc_on_intersection": float(type_acc_on_intersection),
        "type_support_on_intersection": float(type_support_on_intersection),

        # skills-only (name only)
        "precision_skills_only": float(precision_skills_only),
        "recall_skills_only": float(recall_skills_only),
        "f1_skills_only": float(f1_skills_only),

        # bookkeeping
        "k": float(k),
        "gt_nonzero": float(gt_n),

        "exact_match_skills_only": float(exact_match_skills_only),
        "exact_match_with_type": float(exact_match_with_type),
    }


# ---------------- run name ----------------
def build_run_name(
    k_min: int,
    k_max: int,
    use_gap_rule: bool,
    gap_ratio: float,
    prefilter_threshold: Optional[float],
    keep6_min_score: Optional[float],
) -> str:
    parts = [f"topk_{k_min}_{k_max}"]
    parts.append(f"gap_{gap_ratio:.2f}".replace(".", "p") if use_gap_rule else "nogap")
    if prefilter_threshold is not None:
        parts.append(f"pref_{prefilter_threshold:.2f}".replace(".", "p"))
    if keep6_min_score is not None:
        parts.append(f"keep6_{str(keep6_min_score).replace('.', 'p')}")
    return "__".join(parts)


@torch.no_grad()
def score_all_skills_for_text(
    model: nn.Module,
    tokenizer,
    text: str,
    skills_list: List[str],
    device: str,
    max_length: int = 256,
) -> List[Tuple[str, float, float, float, float, int]]:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])  # [1,S,3]
    probs = torch.softmax(logits, dim=-1)[0]  # [S,3]

    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]

    nonzero = 1.0 - p0
    type_label = (p2 > p1).long() + 1  # 1=IMPLICIT, 2=EXPLICIT

    out: List[Tuple[str, float, float, float, float, int]] = []
    for s, nz, a0, a1, a2, tl in zip(skills_list, nonzero, p0, p1, p2, type_label):
        out.append((s, float(nz.item()), float(a0.item()), float(a1.item()), float(a2.item()), int(tl.item())))
    return out


def main() -> None:
    default_splits = REPO_ROOT / "data" / "splits_v1"
    default_skills_file = default_splits / "skills_used.txt"
    default_data_val = default_splits / "val.jsonl"
    default_data_test = default_splits / "test.jsonl"

    default_k_min = 3
    default_k_max = 6
    default_use_gap_rule = True
    default_gap_ratio = 0.25
    default_prefilter_threshold: Optional[float] = None
    default_keep6_min_score: Optional[float] = 0.99

    default_run_name = build_run_name(
        default_k_min,
        default_k_max,
        default_use_gap_rule,
        default_gap_ratio,
        default_prefilter_threshold,
        default_keep6_min_score,
    )

    default_model_dir = find_latest_model_final(ONEPASS_MODEL_BASE)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=str(default_model_dir), help="Path to trained onepass .../final")
    ap.add_argument("--skills_file", type=str, default=str(default_skills_file))
    ap.add_argument("--split_name", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--data_jsonl", type=str, default=str(default_data_val))
    ap.add_argument("--data_jsonl_test", type=str, default=str(default_data_test))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--k_min", type=int, default=default_k_min)
    ap.add_argument("--k_max", type=int, default=default_k_max)

    ap.add_argument("--use_gap_rule", dest="use_gap_rule", action="store_true", default=default_use_gap_rule)
    ap.add_argument("--no_gap_rule", dest="use_gap_rule", action="store_false")

    ap.add_argument("--gap_ratio", type=float, default=default_gap_ratio)
    ap.add_argument("--prefilter_threshold", type=float, default=default_prefilter_threshold)

    ap.add_argument("--keep6_min_score", type=float, default=default_keep6_min_score)
    ap.add_argument("--run_name", type=str, default=default_run_name)

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    skills_path = Path(args.skills_file)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not skills_path.exists():
        raise FileNotFoundError(f"skills_file not found: {skills_path}")

    data_path = Path(args.data_jsonl) if args.split_name == "val" else Path(args.data_jsonl_test)
    if not data_path.exists():
        raise FileNotFoundError(f"data_jsonl not found: {data_path}")

    trained_models_dir = REPO_ROOT / "trained_models" / "one_pass"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = (trained_models_dir / "test_onepass") if args.split_name == "test" else (trained_models_dir / "eval_onepass")
    out_dir = base_dir / args.run_name / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pred_full_jsonl = out_dir / f"predictions_{args.split_name}.jsonl"
    out_mean_metrics = out_dir / f"mean_metrics_{args.split_name}.json"
    out_run_config = out_dir / "run_config.json"
    out_pred_clean_jsonl = out_dir / "predictions_test_clean.jsonl"

    skills_list = [ln.strip() for ln in skills_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not skills_list:
        raise ValueError("skills_used.txt is empty.")

    model, tokenizer, model_name = load_onepass_from_final(
        final_dir=model_dir,
        num_skills=len(skills_list),
        device=args.device,
    )

    items = read_jsonl(data_path)
    n = len(items)

    print("[REPO_ROOT]", REPO_ROOT)
    print("[SPLIT]", args.split_name, "items=", n)
    print("[MODEL_DIR]", model_dir)
    print("[MODEL_NAME]", model_name)
    print("[SKILLS]", len(skills_list))
    print("[DEVICE]", args.device)
    print("[TOPK]", f"{args.k_min}..{args.k_max}", "gap=", args.use_gap_rule, "gap_ratio=", args.gap_ratio)

    preds_rows: List[Dict[str, Any]] = []

    agg = {
        "typed_precision_at_k": 0.0,
        "typed_recall_at_k": 0.0,
        "typed_f1_at_k": 0.0,
        "typed_correct_count": 0.0,

        "type_acc_on_intersection": 0.0,
        "type_support_on_intersection": 0.0,

        "precision_skills_only": 0.0,
        "recall_skills_only": 0.0,
        "f1_skills_only": 0.0,

        "k": 0.0,
        "gt_nonzero": 0.0,

        "exact_match_skills_only": 0.0,
        "exact_match_with_type": 0.0,
    }

    for idx, item in enumerate(items, start=1):
        ex_id = get_id(item)
        text = get_text(item)
        gt = get_gt_skills(item)

        scored = score_all_skills_for_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            skills_list=skills_list,
            device=args.device,
            max_length=args.max_length,
        )

        picked = pick_topk_3_to_6_onepass(
            scored,
            k_max=args.k_max,
            k_min=args.k_min,
            gap_ratio=args.gap_ratio,
            use_gap_rule=args.use_gap_rule,
            prefilter_threshold=args.prefilter_threshold,
            keep6_min_score=args.keep6_min_score,
        )

        if len(picked) < args.k_min:
            picked = pick_topk_3_to_6_onepass(
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

    print("\n=== JD-level MEAN metrics (OnePass, TYPED) ===")
    for kk, v in mean.items():
        print(f"{kk}: {v:.4f}")

    write_jsonl(out_pred_full_jsonl, preds_rows)
    write_json(out_mean_metrics, mean)

    if args.split_name == "test":
        clean_rows = [
            {"id": r["id"], "job_description": r["job_description"], "predicted_skills": r["predicted_skills"]}
            for r in preds_rows
        ]
        write_jsonl(out_pred_clean_jsonl, clean_rows)

    run_config = {
        "repo_root": str(REPO_ROOT),
        "model_dir": str(model_dir),
        "model_name": model_name,
        "data_jsonl": str(data_path),
        "skills_file": str(skills_path),
        "device": args.device,
        "max_length": args.max_length,
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
