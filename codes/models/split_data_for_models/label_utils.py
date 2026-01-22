from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def load_skill_list(skills_path: Path) -> List[str]:
    """
    Load skills from a txt file (one skill per line), remove empty lines & duplicates, keep order.
    """
    if not skills_path.exists():
        raise FileNotFoundError(f"Skills file not found: {skills_path}")

    skills: List[str] = []
    seen = set()

    for line in skills_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        skills.append(s)

    if not skills:
        raise ValueError(f"Skill list is empty: {skills_path}")

    return skills


def sparse_to_dense_labels(skills_sparse: Dict[str, float], skill_list: List[str]) -> List[int]:
    """
    sparse: {skill: 0.5/1.0}  -> dense labels length=len(skill_list) with {0,1,2}
      0 = NONE
      1 = IMPLICIT (0.5)
      2 = EXPLICIT (1.0)
    """
    idx = {s: i for i, s in enumerate(skill_list)}
    y = [0] * len(skill_list)

    for skill, val in skills_sparse.items():
        if skill not in idx:
            raise KeyError(f"Unknown skill '{skill}' not in skill_list.")
        if val == 0.5:
            y[idx[skill]] = 1
        elif val == 1.0:
            y[idx[skill]] = 2
        else:
            raise ValueError(f"Bad label value for '{skill}': {val} (expected 0.5 or 1.0)")
    return y


def dense_to_sparse_skills(y: List[int], skill_list: List[str]) -> Dict[str, float]:
    """
    dense labels {0,1,2} -> sparse {skill: 0.5/1.0} (non-zero only)
    """
    if len(y) != len(skill_list):
        raise ValueError(f"y length {len(y)} != skill_list length {len(skill_list)}")

    out: Dict[str, float] = {}
    for i, cls in enumerate(y):
        if cls == 0:
            continue
        if cls == 1:
            out[skill_list[i]] = 0.5
        elif cls == 2:
            out[skill_list[i]] = 1.0
        else:
            raise ValueError(f"Bad dense label {cls} at index {i} (expected 0/1/2)")
    return out


def stable_example_id(job_description: str, skills_sparse: Dict[str, float]) -> str:
    """
    Deterministic id for items that don't have 'id'.
    """
    payload = {
        "job_description": job_description,
        "skills": dict(sorted(skills_sparse.items(), key=lambda kv: kv[0])),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
