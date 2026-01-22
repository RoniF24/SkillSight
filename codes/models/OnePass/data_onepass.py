from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


# ---------- IO ----------
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
        raise ValueError(f"skills_used is empty: {skills_used_path}")
    return skills


# ---------- Label building ----------
def build_label_vector(
    skills_dict: Dict[str, float],
    skill2idx: Dict[str, int],
    num_skills: int,
) -> torch.Tensor:
    """
    Returns LongTensor[num_skills] with values:
      0 = NONE
      1 = IMPLICIT
      2 = EXPLICIT

    Dataset expected:
      implicit = 0.5
      explicit = 1.0
    """
    y = torch.zeros(num_skills, dtype=torch.long)

    for skill, val in (skills_dict or {}).items():
        idx = skill2idx.get(skill)
        if idx is None:
            continue

        # explicit
        if val == 1.0 or val == 2 or val == 2.0:
            y[idx] = 2
        # implicit
        elif val == 0.5 or val == 1 or val == 1.0:  # tolerate 1 for implicit if it appears
            if val == 1.0:
                # if someone stored explicit as 1.0, it was caught above
                pass
            y[idx] = 1
        # else leave as 0

    return y


# ---------- Dataset ----------
class OnePassDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        tokenizer_name: str,
        skills_used_path: Path,
        max_length: int = 256,
        text_key: str = "job_description",
        skills_key: str = "skills",
    ):
        self.items = items
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

        self.skills = load_skill_list(skills_used_path)
        self.skill2idx = {s: i for i, s in enumerate(self.skills)}

        self.text_key = text_key
        self.skills_key = skills_key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.items[i]

        if self.text_key not in ex:
            raise KeyError(f"Missing '{self.text_key}' in example keys={list(ex.keys())}")

        text = ex[self.text_key]
        skills_dict = ex.get(self.skills_key, {}) or {}
        if not isinstance(skills_dict, dict):
            raise TypeError(f"Expected '{self.skills_key}' to be dict, got {type(skills_dict)}")

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # DataCollator will pad
        )

        labels = build_label_vector(skills_dict, self.skill2idx, len(self.skills))

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,  # LongTensor[num_skills]
        }


# ---------- Dataloaders ----------
@dataclass
class OnePassData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    skills: List[str]


def get_onepass_dataloaders(
    splits_dir: Path,
    tokenizer_name: str = "roberta-base",
    batch_size: int = 8,
    max_length: int = 256,
    num_workers: int = 0,
) -> OnePassData:
    """
    Expects:
      splits_dir/train.jsonl
      splits_dir/val.jsonl
      splits_dir/test.jsonl
      splits_dir/skills_used.txt
    """
    splits_dir = Path(splits_dir)
    skills_used_path = splits_dir / "skills_used.txt"

    train_items = read_jsonl(splits_dir / "train.jsonl")
    val_items = read_jsonl(splits_dir / "val.jsonl")
    test_items = read_jsonl(splits_dir / "test.jsonl")

    train_ds = OnePassDataset(train_items, tokenizer_name, skills_used_path, max_length=max_length)
    val_ds = OnePassDataset(val_items, tokenizer_name, skills_used_path, max_length=max_length)
    test_ds = OnePassDataset(test_items, tokenizer_name, skills_used_path, max_length=max_length)

    collator = DataCollatorWithPadding(tokenizer=train_ds.tokenizer)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded = collator([{k: v for k, v in b.items() if k != "labels"} for b in batch])
        padded["labels"] = torch.stack([b["labels"] for b in batch], dim=0)  # [B, S]
        return padded

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    return OnePassData(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, skills=train_ds.skills)


if __name__ == "__main__":
    # data is next to codes/, so use a relative path from repo root (where you run the command)
    data = get_onepass_dataloaders(Path("data") / "splits_v1")
    batch = next(iter(data.train_loader))
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"].shape)  # [B, S]
    print("num_skills:", len(data.skills))
