"""codes/models/OnePass/train_onepass.py

Trainer script for the OnePass model.

High-level flow:
- Resolve the dataset split directory (`data/splits_v1` by default).
- Load train/val JSONL + the closed skill list (`skills_used.txt`).
- Train a text-only classifier that predicts a 3-way label per skill:
    0 = NONE, 1 = IMPLICIT, 2 = EXPLICIT
- Optionally apply class-weighted cross entropy to reduce collapse to the majority NONE class.
- Save run artifacts (config, metrics, model weights) under `<repo>/<models_dir>/<run_name>/`.

Data pipeline and label building live in `data_onepass.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --- make local imports work no matter where you run from ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_onepass import read_jsonl, load_skill_list, OnePassDataset  # noqa: E402
from metrics_onepass import compute_metrics_onepass_flat  # noqa: E402


def find_repo_root() -> Path:
    """Find repo root by walking up until we find 'codes' and either 'data' or 'DATA'."""
    start = Path(__file__).resolve()
    for p in [start.parent] + list(start.parents):
        # Be tolerant to `data/` vs `DATA/` on different machines.
        if (p / "codes").exists() and ((p / "data").exists() or (p / "DATA").exists()):
            return p
    return Path.cwd().resolve()


def resolve_data_like_path(repo: Path, rel: str) -> Path:
    """
    If rel starts with data/... but repo uses DATA/... (or vice versa), resolve it.
    """
    cand = (repo / rel).resolve()
    if cand.exists():
        return cand

    parts = Path(rel).parts
    if not parts:
        return cand

    first = parts[0]
    if first.lower() == "data":
        alt_first = "DATA" if first != "DATA" else "data"
        alt = (repo / Path(alt_first, *parts[1:])).resolve()
        if alt.exists():
            return alt

    return cand


def make_training_arguments(out_dir: Path, args) -> TrainingArguments:
    """
    transformers changed arg name across versions:
    - older: evaluation_strategy
    - newer: eval_strategy
    We support both with a try/fallback.
    """
    common = dict(
        output_dir=str(out_dir),
        seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        # Correct key (no "eval_" prefix): comes from `compute_metrics_onepass_flat`.
        metric_for_best_model="any_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    try:
        return TrainingArguments(
            **common,
            eval_strategy="epoch",
        )
    except TypeError:
        return TrainingArguments(
            **common,
            evaluation_strategy="epoch",
        )


class OnePassModel(nn.Module):
    """
    OnePass:
      input: text only
      output: logits per skill over 3 classes (0 NONE, 1 IMPLICIT, 2 EXPLICIT)

    logits shape: [B, S, 3]
    labels shape: [B, S]
    """

    def __init__(self, model_name: str, num_skills: int, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.num_skills = num_skills
        self.num_classes = 3

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        hidden = self.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_skills * self.num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, H]
        logits = self.head(self.drop(cls)).view(-1, self.num_skills, self.num_classes)  # [B, S, 3]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, 3), labels.reshape(-1))

        return {"loss": loss, "logits": logits}


class OnePassCollator:
    """
    Pairwise-style collator:
    - uses HF DataCollatorWithPadding for text tensors
    - stacks labels to [B, S]
    """

    def __init__(self, tokenizer):
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        padded = self.inner([{k: v for k, v in b.items() if k != "labels"} for b in batch])
        padded["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        return padded


class WeightedCETrainer(Trainer):
    """
    Trainer with class-weighted CrossEntropyLoss to reduce collapse to majority class (NONE=0).
    Works for OnePass by flattening [B,S,3] -> [N,3] and [B,S] -> [N].
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")  # [B,S,3]

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.reshape(-1, 3), labels.reshape(-1))

        return (loss, outputs) if return_outputs else loss


def compute_class_weights_from_items(items, skill_list) -> Dict[str, object]:
    """
    Compute inverse-frequency class weights for labels {0,1,2} on TRAIN ONLY.
    JSONL uses 0/0.5/1.0.

    We count NONE also for skills missing from each example (full view),
    so NONE will dominate - class weights help.
    """
    counts = Counter()
    for ex in items:
        skills = ex.get("skills", {}) or {}
        present = set()

        for s, v in skills.items():
            if s not in skill_list:
                continue
            present.add(s)
            if float(v) == 1.0:
                counts[2] += 1  # explicit
            elif float(v) == 0.5:
                counts[1] += 1  # implicit
            else:
                counts[0] += 1  # treat anything else as none

        # add NONE for all skills not present
        counts[0] += (len(skill_list) - len(present))

    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    n2 = counts.get(2, 0)
    total = n0 + n1 + n2

    eps = 1e-8
    freq = np.array([n0, n1, n2], dtype=np.float64) / max(total, 1)
    inv = 1.0 / np.maximum(freq, eps)
    inv = inv / inv.mean()
    weights = torch.tensor(inv, dtype=torch.float32)

    return {
        "counts": {"none_0": int(n0), "implicit_1": int(n1), "explicit_2": int(n2), "total": int(total)},
        "weights": weights,
        "weights_list": inv.tolist(),
    }


def save_onepass_hf_config(final_dir: Path, model_name: str, num_skills: int) -> None:
    """
    OnePassModel is plain nn.Module, so trainer.save_model() won't create config.json.
    We manually save a HF config.json (so AutoConfig.from_pretrained(final_dir) works),
    plus extra metadata for our head.
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = AutoConfig.from_pretrained(model_name)
    base_cfg.to_json_file(str(final_dir / "config.json"))

    meta = {
        "onepass": True,
        "base_model_name": model_name,
        "num_skills": int(num_skills),
        "num_classes": 3,
        "label_mapping": {"0": "NONE", "1": "IMPLICIT", "2": "EXPLICIT"},
    }
    (final_dir / "onepass_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits_dir", type=str, default="data/splits_v1")
    ap.add_argument("--model_name", type=str, default="roberta-base")

    # Output folder name is intentionally outside `codes/` (like the pairwise trainer).
    ap.add_argument("--models_dir", type=str, default="trained onepass")  # requested folder name
    ap.add_argument("--run_name", type=str, default="seed42_baseline")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)

    ap.add_argument("--use_class_weights", action="store_true", help="Use class-weighted CE loss (recommended).")

    args = ap.parse_args()

    set_seed(args.seed)

    repo = find_repo_root()

    # Resolve `data/...` vs `DATA/...` depending on the repo layout.
    splits_dir = resolve_data_like_path(repo, args.splits_dir)
    if not splits_dir.exists():
        raise FileNotFoundError(f"splits_dir not found: {splits_dir}")

    # Save under `<repo>/<models_dir>/<run_name>` (no extra `one_pass/` subfolder).
    out_dir = (repo / args.models_dir / args.run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    skills_used = splits_dir / "skills_used.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing: {val_path}")
    if not skills_used.exists():
        raise FileNotFoundError(f"Missing: {skills_used}")

    train_items = read_jsonl(train_path)
    val_items = read_jsonl(val_path)
    skill_list = load_skill_list(skills_used)
    num_skills = len(skill_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = OnePassDataset(
        train_items,
        tokenizer_name=args.model_name,
        skills_used_path=skills_used,
        max_length=args.max_length,
    )
    val_ds = OnePassDataset(
        val_items,
        tokenizer_name=args.model_name,
        skills_used_path=skills_used,
        max_length=args.max_length,
    )

    print("[SPLITS_DIR]", splits_dir)
    print("[TRAIN_PATH]", train_path, "exists=", train_path.exists())
    print("[VAL_PATH]", val_path, "exists=", val_path.exists())
    print("[NUM_TRAIN_ITEMS]", len(train_items))
    print("[NUM_VAL_ITEMS]", len(val_items))
    print("[NUM_SKILLS]", len(skill_list))
    print("[LEN(train_ds)]", len(train_ds))
    print("[LEN(val_ds)]", len(val_ds))

    ex0 = train_ds[0]
    if "attention_mask" in ex0:
        am = ex0["attention_mask"]
        tok_len0 = int(am.sum().item()) if torch.is_tensor(am) else int(sum(am))
    else:
        ids = ex0["input_ids"]
        tok_len0 = int(len(ids))
    print("[EX0 token_len]", tok_len0, "labels_shape=", tuple(ex0["labels"].shape))

    collator = OnePassCollator(tokenizer=tokenizer)
    model = OnePassModel(model_name=args.model_name, num_skills=num_skills)

    cw_info = compute_class_weights_from_items(train_items, skill_list)
    print("[TRAIN LABEL COUNTS]", cw_info["counts"])
    print("[CLASS WEIGHTS (normalized)]", cw_info["weights_list"])

    run_config = {
        "model_type": "one_pass",
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "max_length": args.max_length,
        "num_skills": num_skills,
        "splits_dir": str(splits_dir),
        "run_name": args.run_name,
        "output_dir": str(out_dir),
        "use_class_weights": bool(args.use_class_weights),
        "train_label_counts": cw_info["counts"],
        "class_weights": cw_info["weights_list"],
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    def hf_compute_metrics(eval_pred) -> Dict[str, float]:
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        return compute_metrics_onepass_flat((logits, labels))

    targs = make_training_arguments(out_dir, args)

    if args.use_class_weights:
        trainer = WeightedCETrainer(
            class_weights=cw_info["weights"],
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=hf_compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=hf_compute_metrics,
        )

    train_out = trainer.train()

    trainer.save_metrics("train", train_out.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    history_path = out_dir / "metrics_history.jsonl"
    with history_path.open("w", encoding="utf-8") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save final model (weights) + tokenizer + config.json + onepass_meta.json
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    save_onepass_hf_config(final_dir, args.model_name, num_skills)

    print("[OK] Saved model to:", final_dir)


if __name__ == "__main__":
    main()
