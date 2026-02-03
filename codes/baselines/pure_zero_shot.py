from __future__ import annotations

import json
import os
import sys
import time
import random
import socket
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2] if len(THIS_FILE.parents) >= 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PATH = REPO_ROOT / "data" / "synthetic_dataset.jsonl"
SKILLS_PATH = REPO_ROOT / "data" / "skills_v1.txt"
PROMPT_PATH = REPO_ROOT / "codes" / "prompts" / "zero_shot_baseline.txt"
OUT_DIR = REPO_ROOT / "baselines_outputs"

VERBOSE = False


@dataclass
class Args:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_examples: int = 3500
    seed: Optional[int] = None
    max_retries: int = 8
    backoff_cap_s: int = 60


def parse_args(argv: List[str]) -> Args:
    try:
        import argparse as _argparse
    except Exception:
        _argparse = None

    a = Args()
    if _argparse is None:
        return a

    ap = _argparse.ArgumentParser(
        description="Zero-shot baseline with OpenAI: per-skill explicit/implicit scores from job descriptions."
    )
    ap.add_argument("--model", type=str, default=a.model)
    ap.add_argument("--temperature", type=float, default=a.temperature)
    ap.add_argument("--max-examples", type=int, default=a.max_examples)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=a.max_retries, help="Retries on transient HTTP errors (429/5xx).")
    ap.add_argument("--backoff-cap-s", type=int, default=a.backoff_cap_s, help="Max sleep seconds between retries.")

    ns = ap.parse_args(argv)
    return Args(
        model=ns.model,
        temperature=ns.temperature,
        max_examples=ns.max_examples,
        seed=ns.seed,
        max_retries=ns.max_retries,
        backoff_cap_s=ns.backoff_cap_s,
    )


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no}: {e}") from e
    return rows


def load_global_skills(path: Path) -> List[str]:
    skills: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            skills.append(s)
    return skills


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def make_out_paths(model: str) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"zero_shot_{model.replace(':', '_')}_{ts}"
    preds_path = OUT_DIR / f"{base}_predictions.jsonl"
    summary_path = OUT_DIR / f"{base}_summary.json"
    return preds_path, summary_path


def call_openai(
    prompt: str,
    model: str,
    temperature: float,
    *,
    max_retries: int = 8,
    backoff_cap_s: int = 60,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = base + "/responses"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "temperature": temperature,
        "max_output_tokens": 1200,
        # JSON mode to force valid JSON output (Responses API)
        "text": {"format": {"type": "json_object"}},
    }

    data = json.dumps(payload).encode("utf-8")

    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                obj = json.loads(resp.read().decode("utf-8", errors="replace"))
            # success -> parse
            break

        except urllib.error.HTTPError as e:
            last_err = e
            code = getattr(e, "code", None)

            # transient errors -> retry
            if code in (429, 500, 502, 503, 504):
                sleep_s = min(backoff_cap_s, 2 ** attempt)
                print(
                    f"[WARN] OpenAI HTTP {code} (attempt {attempt}/{max_retries}). "
                    f"Retrying in {sleep_s}s...",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue

            # non-transient -> raise immediately
            raise

        except (urllib.error.URLError, socket.timeout) as e:
            last_err = e
            sleep_s = min(backoff_cap_s, 2 ** attempt)
            print(
                f"[WARN] OpenAI network/timeout error (attempt {attempt}/{max_retries}). "
                f"Retrying in {sleep_s}s... ({type(e).__name__})",
                flush=True,
            )
            time.sleep(sleep_s)
            continue

    else:
        raise RuntimeError(f"OpenAI request failed after {max_retries} retries: {last_err}")

    texts: List[str] = []
    for item in obj.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    t = part.get("text")
                    if isinstance(t, str):
                        texts.append(t)

    if texts:
        return "\n".join(texts).strip()

    ot = obj.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()
    return ""


def build_prompt(template: str, job_description: str, global_skills: List[str]) -> str:
    skills_block = "\n".join(f"- {s}" for s in global_skills)
    return template.format(JOB_DESCRIPTION=job_description, GLOBAL_SKILLS=skills_block)


def parse_predictions(raw: str, global_skills: List[str]) -> Dict[str, Any]:
    text = raw.strip()

    def try_json_any(t: str) -> Optional[Any]:
        try:
            return json.loads(t)
        except Exception:
            pass
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except Exception:
                pass
        return None

    obj = try_json_any(text)

    global_lower = {s.lower(): s for s in global_skills}
    skills_scores: Dict[str, float] = {}

    def add_skill(name: str, val: Any) -> None:
        if not isinstance(name, str):
            return
        name = name.strip().lstrip("-*•").strip()
        if not name:
            return
        key = name.lower()
        if key not in global_lower:
            return

        # accept "explicit"/"implicit" too
        if isinstance(val, str):
            vlow = val.strip().lower()
            if vlow in ("1", "explicit"):
                val = 1.0
            elif vlow in ("0.5", "implicit"):
                val = 0.5
            else:
                return

        try:
            v = float(val)
        except Exception:
            return

        # snap to {0.5,1}
        if v >= 0.75:
            v = 1.0
        elif v >= 0.25:
            v = 0.5
        else:
            return  # omit 0

        skills_scores[global_lower[key]] = v

    if isinstance(obj, dict):
        # expected: {"skills": {...}}
        if "skills" in obj and isinstance(obj["skills"], dict):
            for k, v in obj["skills"].items():
                add_skill(k, v)
        else:
            # also accept direct map {"Skill": 1, ...}
            for k, v in obj.items():
                if k == "skills":
                    continue
                add_skill(k, v)

    explicit = [s for s, v in skills_scores.items() if v >= 0.99]
    implicit = [s for s, v in skills_scores.items() if 0.49 <= v < 0.99]
    all_list = sorted(set(explicit) | set(implicit))

    return {
        "explicit": explicit,
        "implicit": implicit,
        "all": all_list,
        "skills_scores": skills_scores,
    }


def get_gold_labels(skills: Dict[str, float], scope: str) -> List[str]:
    if scope == "explicit":
        return [s for s, w in skills.items() if w >= 0.99]
    elif scope == "implicit":
        return [s for s, w in skills.items() if 0.49 <= w < 0.99]
    elif scope == "all":
        return [s for s, w in skills.items() if w >= 0.49]
    else:
        raise ValueError(f"Unknown scope: {scope}")


def compute_metrics(gold: List[str], pred: List[str]) -> Tuple[float, float, float, List[str]]:
    gold_set = set(gold)
    pred_set = set(pred)
    overlap = sorted(gold_set & pred_set)

    prec = 0.0 if not pred_set else len(overlap) / len(pred_set)
    rec = 0.0 if not gold_set else len(overlap) / len(gold_set)
    f1 = 0.0 if (prec + rec) == 0.0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1, overlap


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.seed is not None:
        random.seed(args.seed)

    rows = load_dataset(DATA_PATH)
    global_skills = load_global_skills(SKILLS_PATH)
    template = load_prompt_template(PROMPT_PATH)

    preds_path, summary_path = make_out_paths(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_f = preds_path.open("w", encoding="utf-8")

    max_n = min(args.max_examples, len(rows))
    scopes = ["explicit", "implicit", "all"]
    per_example_metrics: List[Dict[str, Dict[str, float]]] = []

    print(f"[INFO] Loaded {len(rows)} examples, limiting to first {max_n}\n")

    for idx, row in enumerate(rows[:max_n], start=1):
        job = row.get("job_description", "")
        gold_skills_dict: Dict[str, float] = row.get("skills", {}) or {}

        remaining = max_n - idx
        print(f"[PROGRESS] example {idx}/{max_n}  (remaining: {remaining})", flush=True)

        prompt = build_prompt(template, job, global_skills)
        raw = call_openai(
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            backoff_cap_s=args.backoff_cap_s,
        )
        parsed = parse_predictions(raw, global_skills)

        pred_explicit = parsed["explicit"]
        pred_implicit = parsed["implicit"]
        pred_all = parsed["all"]
        skills_scores = parsed["skills_scores"]

        metrics_by_scope: Dict[str, Dict[str, Any]] = {}
        for scope in scopes:
            gold_labels = get_gold_labels(gold_skills_dict, scope)
            if scope == "explicit":
                pred_list = pred_explicit
            elif scope == "implicit":
                pred_list = pred_implicit
            else:
                pred_list = pred_all

            prec, rec, f1, overlap = compute_metrics(gold_labels, pred_list)
            metrics_by_scope[scope] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "gold": gold_labels,
                "pred": pred_list,
                "overlap": overlap,
            }

        per_example_metrics.append(
            {
                scope: {
                    "precision": metrics_by_scope[scope]["precision"],
                    "recall": metrics_by_scope[scope]["recall"],
                    "f1": metrics_by_scope[scope]["f1"],
                }
                for scope in scopes
            }
        )

        example_row = {
            "job_description": job,
            "skills": skills_scores,
        }
        pred_f.write(json.dumps(example_row, ensure_ascii=False) + "\n")
        pred_f.flush()

    pred_f.close()

    n = len(per_example_metrics) if per_example_metrics else 1
    avg_by_scope: Dict[str, Dict[str, float]] = {}
    for scope in scopes:
        avg_by_scope[scope] = {
            "precision": sum(m[scope]["precision"] for m in per_example_metrics) / n,
            "recall": sum(m[scope]["recall"] for m in per_example_metrics) / n,
            "f1": sum(m[scope]["f1"] for m in per_example_metrics) / n,
        }

    summary = {
        "model": args.model,
        "num_examples": n,
        "avg_metrics": avg_by_scope,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(DATA_PATH),
        "skills_path": str(SKILLS_PATH),
        "prompt_path": str(PROMPT_PATH),
        "max_retries": args.max_retries,
        "backoff_cap_s": args.backoff_cap_s,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY]")
    for scope in scopes:
        s = avg_by_scope[scope]
        print(f"{scope.upper():>8}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")

    print(f"\nPredictions saved to: {preds_path}")
    print(f"Summary saved to:     {summary_path}")


if __name__ == "__main__":
    main()
