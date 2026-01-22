from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
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

from data_pairwise import read_jsonl, load_skill_list, build_pair_examples, PairwiseDataset  # noqa: E402
from metrics_pairwise import compute_metrics_pairwise  # noqa: E402


def find_repo_root() -> Path:
    """
    Find repo root by walking up from THIS FILE location until we find both 'codes' and 'data'.
    """
    start = Path(__file__).resolve()
    for p in [start.parent] + list(start.parents):
        if (p / "codes").exists() and (p / "data").exists():
            return p
    return Path.cwd().resolve()


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


class WeightedCETrainer(Trainer):
    """
    Trainer with class-weighted CrossEntropyLoss to reduce collapse to majority class (NONE=0).
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_pairs) -> Dict[str, object]:
    """
    Compute inverse-frequency class weights for labels {0,1,2}.
    Returns: counts + weights (torch tensor)
    """
    counts = Counter([ex.label for ex in train_pairs])
    # ensure all classes exist in dict
    n0 = counts.get(0, 0)
    n1 = counts.get(1, 0)
    n2 = counts.get(2, 0)

    total = n0 + n1 + n2
    # Avoid division by zero (if a class is missing)
    eps = 1e-8
    freq = np.array([n0, n1, n2], dtype=np.float64) / max(total, 1)
    inv = 1.0 / np.maximum(freq, eps)

    # Normalize weights so average weight ~ 1 (nice for stability)
    inv = inv / inv.mean()

    weights = torch.tensor(inv, dtype=torch.float32)

    return {
        "counts": {"none_0": n0, "implicit_1": n1, "explicit_2": n2, "total": total},
        "weights": weights,
        "weights_list": inv.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--splits_dir", type=str, default="data/splits_v1")
    ap.add_argument("--model_name", type=str, default="roberta-base")
    ap.add_argument("--output_dir", type=str, default="trained pairwise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--min_negs", type=int, default=5)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)

    ap.add_argument("--use_class_weights", action="store_true", help="Use class-weighted CE loss (recommended).")

    ap.add_argument(
        "--eval_full",
        action="store_true",
        help="Evaluate val on ALL skills (slower, more realistic).",
    )
    args = ap.parse_args()

    # Reproducibility
    set_seed(args.seed)

    repo = find_repo_root()
    splits_dir = (repo / args.splits_dir).resolve()
    out_dir = (repo / args.output_dir).resolve()
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_pairs = build_pair_examples(
        train_items,
        skill_list,
        seed=args.seed,
        neg_per_pos=args.neg_per_pos,
        min_negs=args.min_negs,
        eval_full=False,
    )
    val_pairs = build_pair_examples(
        val_items,
        skill_list,
        seed=args.seed,
        neg_per_pos=args.neg_per_pos,
        min_negs=args.min_negs,
        eval_full=bool(args.eval_full),
    )

    # ---- log class distribution + (optional) class weights ----
    cw_info = compute_class_weights(train_pairs)
    print("[TRAIN LABEL COUNTS]", cw_info["counts"])
    print("[CLASS WEIGHTS (normalized)]", cw_info["weights_list"])

    run_config = {
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "max_length": args.max_length,
        "neg_per_pos": args.neg_per_pos,
        "min_negs": args.min_negs,
        "eval_full": bool(args.eval_full),
        "use_class_weights": bool(args.use_class_weights),
        "train_label_counts": cw_info["counts"],
        "class_weights": cw_info["weights_list"],
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    train_ds = PairwiseDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_ds = PairwiseDataset(val_pairs, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def hf_compute_metrics(eval_pred) -> Dict[str, float]:
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids

        preds = np.argmax(logits, axis=-1).tolist()
        y_true = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        return compute_metrics_pairwise(y_true=y_true, y_pred=preds)

    targs = make_training_arguments(out_dir, args)

    # Choose trainer class
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

    # ---- Train ONCE (important!) ----
    train_out = trainer.train()

    # Save metrics + state
    trainer.save_metrics("train", train_out.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", eval_metrics)

    # Full history
    history_path = out_dir / "metrics_history.jsonl"
    with history_path.open("w", encoding="utf-8") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save final model
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[OK] Saved model to:", final_dir)


if __name__ == "__main__":
    main()
