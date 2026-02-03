from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# --- local import (label_utils.py in the same folder) ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from label_utils import load_skill_list, dense_to_sparse_skills, stable_example_id  # noqa: E402


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking up until a folder contains BOTH 'codes' and 'data'.
    This matches your layout: codes/... and data/... at the project root.
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "codes").exists() and (p / "data").exists():
            return p
    # fallback: if not found, assume 4 levels up (won't usually happen)
    return start.resolve().parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def resolve_from_root(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path} at line {line_no}: {e}") from e
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_item(item: Dict[str, Any], skill_list: List[str]) -> Dict[str, Any]:
    """
    Accepts either:
      A) {id?, job_description/text, skills:{skill:0.5/1.0}}
      B) {id?, job_description/text, y:[0/1/2,...]}  (legacy)
    Outputs canonical:
      {id, job_description, skills:{skill:0.5/1.0}}  (skills are sparse: non-zero only)
    """
    jd = item.get("job_description", item.get("text"))
    if not isinstance(jd, str) or not jd.strip():
        raise ValueError("Missing or empty 'job_description'/'text'")

    if "skills" in item:
        skills = item["skills"]
        if not isinstance(skills, dict):
            raise ValueError("'skills' must be a dict {skill:0.5/1.0}")

        skills_sparse: Dict[str, float] = {}
        for k, v in skills.items():
            if v in (0.5, 1.0):
                skills_sparse[str(k)] = float(v)
            else:
                raise ValueError(f"Bad skills value for '{k}': {v} (expected 0.5 or 1.0)")

    elif "y" in item:
        y = item["y"]
        if not isinstance(y, list):
            raise ValueError("'y' must be a list of 0/1/2")
        y_int = [int(x) for x in y]
        skills_sparse = dense_to_sparse_skills(y_int, skill_list)

    else:
        raise ValueError("Item must contain either 'skills' or 'y'")

    ex_id = item.get("id")
    if not isinstance(ex_id, str) or not ex_id.strip():
        ex_id = stable_example_id(jd, skills_sparse)

    return {"id": ex_id, "job_description": jd, "skills": skills_sparse}


def main() -> None:
    ap = argparse.ArgumentParser()

    # Your layout:
    # - data is outside codes
    # - codes/models and codes/skills are inside codes
    ap.add_argument("--data_path", type=str, default="data/synthetic_dataset.jsonl")
    ap.add_argument("--skills_path", type=str, default="codes/skills/skills_v1.txt")
    ap.add_argument("--out_dir", type=str, default="codes/models/split_data_for_models/splits_v1")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-9:
        raise ValueError("Ratios must sum to 1.0")

    data_path = resolve_from_root(args.data_path)
    skills_path = resolve_from_root(args.skills_path)
    out_dir = resolve_from_root(args.out_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not skills_path.exists():
        raise FileNotFoundError(f"Skills file not found: {skills_path}")

    skill_list = load_skill_list(skills_path)

    raw_items = read_jsonl(data_path)
    if not raw_items:
        raise ValueError(f"No items found in dataset: {data_path}")

    items = [normalize_item(it, skill_list) for it in raw_items]

    rnd = random.Random(args.seed)
    rnd.shuffle(items)

    n = len(items)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "train.jsonl", train_items)
    write_jsonl(out_dir / "val.jsonl", val_items)
    write_jsonl(out_dir / "test.jsonl", test_items)

    # snapshot for reproducibility
    (out_dir / "skills_used.txt").write_text("\n".join(skill_list) + "\n", encoding="utf-8")

    split_info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "repo_root": str(REPO_ROOT),
        "source_data": str(data_path),
        "skills_source": str(skills_path),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Safety: no overlap in IDs
    def ids(xs: List[Dict[str, Any]]) -> set[str]:
        return {x["id"] for x in xs}

    if (ids(train_items) & ids(val_items)) or (ids(train_items) & ids(test_items)) or (ids(val_items) & ids(test_items)):
        raise RuntimeError("ID overlap detected between splits!")

    print("[OK] Splits created at:", out_dir)
    print("[OK] Counts:", split_info["counts"])


if __name__ == "__main__":
    main()
