from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


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


def load_skill_list(skills_used_path: Path) -> List[str]:
    skills: List[str] = []
    seen = set()
    for line in skills_used_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        skills.append(s)
    if not skills:
        raise ValueError(f"skills_used.txt is empty: {skills_used_path}")
    return skills


def _is_close(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


def skill_score_to_class(score: float) -> int:
    """
    External (dataset) scores: 0/0.5/1.0
    Internal (training) labels: 0/1/2 (NONE/IMPLICIT/EXPLICIT)
    """
    # allow tiny float noise (e.g., 0.5000000001)
    if _is_close(score, 0.5):
        return 1
    if _is_close(score, 1.0):
        return 2
    raise ValueError(f"Bad score {score} (expected 0.5 or 1.0)")


@dataclass(frozen=True)
class PairExample:
    ex_id: str
    skill: str
    text: str
    label: int  # 0/1/2


def build_pair_examples(
    items: List[Dict[str, Any]],
    skill_list: List[str],
    seed: int,
    neg_per_pos: int = 3,
    min_negs: int = 5,
    eval_full: bool = False,
) -> List[PairExample]:
    """
    Train: eval_full=False and we sample negatives.
    Val/Test: optionally eval_full=True to evaluate all skills (accurate but larger).
    """
    rnd = random.Random(seed)
    all_skills = skill_list[:]
    examples: List[PairExample] = []

    for idx_item, it in enumerate(items):
        ex_id = str(it.get("id", "")).strip()
        if not ex_id:
            raise ValueError(
                f"Missing 'id' in item index {idx_item}. "
                "Each sample must have a stable id (hash) for consistent splits."
            )

        text = it.get("job_description", it.get("text", ""))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing job_description/text in item id={ex_id}")

        skills_sparse: Any = it.get("skills", {})
        if not isinstance(skills_sparse, dict):
            raise ValueError(f"skills must be a dict in item id={ex_id}")

        # positives map: skill -> class label (1/2)
        pos_map: Dict[str, int] = {}
        for s, score in skills_sparse.items():
            try:
                sc = float(score)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Bad skill score for skill={s} in item id={ex_id}: {score}") from e

            # IMPORTANT: allow sparse dicts that may include 0.0 (ignore those)
            if _is_close(sc, 0.0):
                continue

            pos_map[str(s)] = skill_score_to_class(sc)

        if eval_full:
            # create ALL skills for this item
            for s in all_skills:
                label = pos_map.get(s, 0)
                examples.append(PairExample(ex_id=ex_id, skill=s, text=text, label=label))
            continue

        # always include all positives
        pos_skills = list(pos_map.keys())
        for s in pos_skills:
            examples.append(PairExample(ex_id=ex_id, skill=s, text=text, label=pos_map[s]))

        # sample negatives
        neg_candidates = [s for s in all_skills if s not in pos_map]
        if neg_candidates:
            n_pos = max(1, len(pos_skills))
            n_negs = max(min_negs, neg_per_pos * n_pos)
            n_negs = min(n_negs, len(neg_candidates))
            negs = rnd.sample(neg_candidates, k=n_negs)
            for s in negs:
                examples.append(PairExample(ex_id=ex_id, skill=s, text=text, label=0))

    return examples


class PairwiseDataset(Dataset):
    def __init__(self, pair_examples: List[PairExample], tokenizer, max_length: int = 256):
        self.pairs = pair_examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.pairs[idx]
        enc = self.tokenizer(
            ex.skill,
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Trainer/DataCollator will pad dynamically
        )
        # IMPORTANT: return labels as int; collator will tensorize properly
        enc["labels"] = int(ex.label)
        return enc
