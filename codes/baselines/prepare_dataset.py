
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any

# נתיבי קבצים
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "synthetic_dataset.jsonl"
SKILLS_PATH = REPO_ROOT / "data" / "skills_v1.txt"


def load_global_skills(path: Path) -> List[str]:
    skills: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            skills.append(line)
    return skills


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e

            if "job_description" not in obj or "skills" not in obj:
                raise ValueError(f"Missing keys on line {line_no}: {obj.keys()}")
            examples.append(obj)
    return examples


def build_label_matrix(
    examples: List[Dict[str, Any]],
    global_skills: List[str],
) -> List[List[float]]:
    """
    ממפה כל דוגמה לוקטור באורך len(global_skills)
    לפי הסקילים שבמילון "skills" של אותה דוגמה.
    """
    label_matrix: List[List[float]] = []
    for ex in examples:
        ex_skills: Dict[str, float] = ex["skills"]
        vec = [float(ex_skills.get(skill, 0.0)) for skill in global_skills]
        label_matrix.append(vec)
    return label_matrix


def main() -> None:
    print(f"[INFO] Loading dataset from: {DATA_PATH}")
    examples = load_dataset(DATA_PATH)
    print(f"[INFO] Loaded {len(examples)} examples")

    print(f"[INFO] Loading global skills from: {SKILLS_PATH}")
    global_skills = load_global_skills(SKILLS_PATH)
    print(f"[INFO] Loaded {len(global_skills)} global skills")

    labels = build_label_matrix(examples, global_skills)
    print(f"[INFO] Built label matrix with shape: {len(labels)} x {len(global_skills)}")

    # הצצה קטנה לדוגמה ראשונה
    if examples:
        print("\n[DEBUG] First example snippet:")
        print("job_description:", examples[0]["job_description"][:140], "...")
        print("non-zero skills for first example:")
        ex0_skills = examples[0]["skills"]
        print(ex0_skills)


if __name__ == "__main__":
    main()
