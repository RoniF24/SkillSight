from __future__ import annotations

"""codes/sampler/make_plans.py

Plan sampler for the synthetic dataset generator.

This script creates a JSONL file of "plans" that later drive text generation. Each plan is a small structured
spec describing which skills must appear explicitly vs implicitly, plus a few metadata fields.

Output schema (one JSON object per line):
- plan_id: running integer id (1..N)
- bundle: which bundle the plan was sampled from
- k: total number of skills in the plan
- explicit: list of skills that must be mentioned verbatim in the generated text (label=1.0 downstream)
- implicit: list of skills that must be suggested indirectly (label=0.5 downstream)
- all_skills: explicit + implicit combined
- domain, seniority: lightweight context fields used by the prompt
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple


# ===== CONFIG (edit here) =====
# Paths are resolved relative to the repo root when you run the script from there.
# Note: the repository's default skill list lives under `data/skills_v1.txt`.
SKILLS_PATH = Path("cdata/skills_v1.txt")
BUNDLES_PATH = Path("data/bundles_v1.json")

OUT_PLANS_PATH = Path("data/plans/plans_v1.jsonl")

# TOTAL_PLANS controls how many plan rows to emit overall (not per bundle).
# Keeping this small is useful for quick sanity checks; scale it up for real dataset generation.
TOTAL_PLANS = 1

# Reproducibility: same SEED => same sampled plan sequence.
SEED = 50

K_MIN = 3
K_MAX = 6

IMPLICIT_RATIO_MIN = 0.60
IMPLICIT_RATIO_MAX = 0.75

DOMAINS = ["FinTech","Healthcare","E-commerce","SaaS","Telecom","EdTech","Cybersecurity","Logistics"]
SENIORITIES = ["Junior", "Mid-level", "Senior"]

# Special rule:
# If k == 6 -> force implicit=4 (0.5) and explicit=2 (1.0).
# This keeps the implicit/explicit split stable for the largest plans.
FORCE_SPLIT_FOR_K6 = (2, 4)  # (explicit, implicit)
# =============================


# ---------- IO helpers ----------

def read_lines(path: Path) -> List[str]:
    """Read non-empty stripped lines from a text file."""
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def load_json(path: Path) -> Any:
    """Load a JSON file into a Python object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _as_str_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _normalize_pick_one_of(p: Any) -> List[List[str]]:
    """
    Normalizes pick_one_of into a list of groups.
    Supported shapes:
      - ["A","B"] -> [["A","B"]]
      - [["A","B"],["C","D"]] -> same
      - {"groups":[...]} / {"options":[...]} -> tries common keys
    """
    groups: List[List[str]] = []

    if not p:
        return groups

    if isinstance(p, list):
        if all(isinstance(g, list) for g in p):
            groups = [[str(x) for x in g] for g in p]
        else:
            groups = [[str(x) for x in p]]
    elif isinstance(p, dict):
        if isinstance(p.get("groups"), list):
            raw = p["groups"]
            if all(isinstance(g, list) for g in raw):
                groups = [[str(x) for x in g] for g in raw]
            else:
                groups = [[str(x) for x in raw]]
        elif isinstance(p.get("options"), list):
            groups = [[str(x) for x in p["options"]]]
        else:
            # fallback: treat dict values as one group
            groups = [[str(x) for x in p.values()]]

    # clean groups
    cleaned: List[List[str]] = []
    for g in groups:
        g2 = _dedupe_keep_order([s for s in g if s])
        if g2:
            cleaned.append(g2)
    return cleaned


def normalize_bundle(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a raw bundle object into a normalized spec:
    {
      "id": "...",
      "must_have": [...],
      "optional": [...],
      "pick_one_of_groups": [[...], [...], ...]
    }
    """
    bid = str(obj.get("id") or obj.get("name") or obj.get("bundle") or obj.get("role") or "unknown_bundle")

    must_have = _dedupe_keep_order(_as_str_list(obj.get("must_have")))
    optional = _dedupe_keep_order(_as_str_list(obj.get("optional")))
    pick_groups = _normalize_pick_one_of(obj.get("pick_one_of"))

    return {
        "id": bid,
        "must_have": must_have,
        "optional": optional,
        "pick_one_of_groups": pick_groups,
    }


def load_bundles(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Expected file shape:
      {"version":"v1","bundles":[{...},{...}]}

    Returns:
      dict bundle_id -> normalized spec
    """
    data = load_json(path)
    if not isinstance(data, dict) or "bundles" not in data or not isinstance(data["bundles"], list):
        raise ValueError(f"Expected dict with 'bundles' list in: {path}")

    out: Dict[str, Dict[str, Any]] = {}
    for obj in data["bundles"]:
        if not isinstance(obj, dict):
            continue
        spec = normalize_bundle(obj)
        out[spec["id"]] = spec

    if not out:
        raise ValueError(f"No bundles parsed from: {path}")
    return out


# ---------- sampling logic ----------

def pick_k(rng: random.Random) -> int:
    """
    Choose number of skills per plan: K_MIN..K_MAX
    Slight bias to 4-5 when range is 3..6.
    """
    choices = list(range(K_MIN, K_MAX + 1))
    if choices == [3, 4, 5, 6]:
        weights = [0.15, 0.35, 0.35, 0.15]
        return rng.choices(choices, weights=weights, k=1)[0]
    return rng.choice(choices)


def split_counts(k: int, rng: random.Random) -> Tuple[int, int, float]:
    """
    Returns (explicit_count, implicit_count, implicit_ratio_target)

    Rules:
    - implicit ratio target is between IMPLICIT_RATIO_MIN..MAX
    - if k == 6: force FORCE_SPLIT_FOR_K6 (your rule)
    - always keep at least 1 explicit
    """
    if k == 6:
        exp, imp = FORCE_SPLIT_FOR_K6
        return exp, imp, imp / k

    implicit_target = rng.uniform(IMPLICIT_RATIO_MIN, IMPLICIT_RATIO_MAX)
    implicit_count = math.ceil(k * implicit_target)

    # ensure at least 1 explicit
    implicit_count = min(implicit_count, k - 1)
    explicit_count = k - implicit_count

    return explicit_count, implicit_count, implicit_target


def sample_skills_from_bundle(spec: Dict[str, Any], rng: random.Random) -> List[str]:
    """
    Build the final chosen skill list:
    - include all must_have
    - choose exactly 1 from each pick_one_of group (if any)
    - fill remaining up to k from optional (or fallback pool if needed)
    """
    must_have: List[str] = list(spec["must_have"])
    optional: List[str] = list(spec["optional"])
    pick_groups: List[List[str]] = list(spec["pick_one_of_groups"])

    # required skills count:
    required_count = len(must_have) + len(pick_groups)

    if required_count > K_MAX:
        raise ValueError(
            f"Bundle '{spec['id']}' requires {required_count} skills (must_have + pick_one_of groups) "
            f"but K_MAX={K_MAX}. Reduce requirements or increase K_MAX."
        )

    # choose k, but ensure it can fit required skills
    k = pick_k(rng)
    k = max(k, required_count)

    chosen: List[str] = []
    chosen.extend(must_have)

    # pick one from each group
    for g in pick_groups:
        choice = rng.choice(g)
        chosen.append(choice)

    chosen = _dedupe_keep_order(chosen)

    # available pools (avoid duplicates)
    chosen_set = set(chosen)

    # primary fill: optional
    opt_pool = [s for s in optional if s not in chosen_set]
    rng.shuffle(opt_pool)

    # fallback fill: from all skills mentioned in bundle (optional + all pick options + must_have)
    fallback_pool: List[str] = []
    fallback_pool.extend(optional)
    for g in pick_groups:
        fallback_pool.extend(g)
    fallback_pool.extend(must_have)
    fallback_pool = _dedupe_keep_order([s for s in fallback_pool if s not in chosen_set])
    rng.shuffle(fallback_pool)

    while len(chosen) < k:
        if opt_pool:
            s = opt_pool.pop()
        elif fallback_pool:
            s = fallback_pool.pop()
        else:
            break
        if s not in chosen_set:
            chosen.append(s)
            chosen_set.add(s)

    # if bundle doesn't have enough unique skills, clamp k to what we could get
    return chosen


def split_explicit_implicit(chosen: List[str], spec: Dict[str, Any], rng: random.Random) -> Tuple[List[str], List[str], float]:
    """
    Prefer MUST_HAVE skills to be explicit, then pick_one_of choices, then others.
    Still respects counts from split_counts().
    """
    k = len(chosen)
    explicit_count, implicit_count, implicit_target = split_counts(k, rng)

    must_have = set(spec["must_have"])

    # preference order: must_have first, then rest
    preferred = [s for s in chosen if s in must_have]
    others = [s for s in chosen if s not in must_have]

    # fill explicit from preferred then others
    explicit: List[str] = []
    for s in preferred:
        if len(explicit) < explicit_count:
            explicit.append(s)

    if len(explicit) < explicit_count:
        # add from others randomly but stable with rng
        pool = list(others)
        rng.shuffle(pool)
        for s in pool:
            if len(explicit) < explicit_count:
                explicit.append(s)

    explicit_set = set(explicit)
    implicit = [s for s in chosen if s not in explicit_set]

    # Sanity checks: these should always hold given how we compute the split.
    assert len(explicit) == explicit_count
    assert len(implicit) == implicit_count

    return explicit, implicit, implicit_target

def pick_seniority_by_k(k: int, rng: random.Random) -> str:
    """Heuristic: plans with more skills tend to map to higher seniority."""
    # weights order must match SENIORITIES
    # SENIORITIES = ["Junior", "Mid-level", "Senior"]
    if k <= 4:
        weights = [0.65, 0.30, 0.05]
    elif k == 5:
        weights = [0.20, 0.60, 0.20]
    else:  # k == 6
        weights = [0.05, 0.35, 0.60]
    return rng.choices(SENIORITIES, weights=weights, k=1)[0]


def sample_plan(bundle_id: str, spec: Dict[str, Any], global_skills: set[str], rng: random.Random) -> Dict[str, Any]:
    chosen = sample_skills_from_bundle(spec, rng)

    # Keep the dataset vocabulary "closed": every sampled skill must appear in the global skill list.
    unknown = [s for s in chosen if s not in global_skills]
    if unknown:
        raise ValueError(f"Bundle '{bundle_id}' produced unknown skills (first 10): {unknown[:10]}")

    explicit, implicit, implicit_target = split_explicit_implicit(chosen, spec, rng)

    domain = rng.choice(DOMAINS)
    seniority = pick_seniority_by_k(len(chosen), rng)

    return {
        "bundle": bundle_id,
        "k": len(chosen),
        "implicit_ratio_target": round(float(implicit_target), 3),
        "explicit": explicit,
        "implicit": implicit,
        "all_skills": chosen,
        "domain": domain,
        "seniority": seniority,
    }


def run() -> None:
    """Generate `TOTAL_PLANS` plan rows and write them as JSONL to `OUT_PLANS_PATH`."""
    global_skills = set(read_lines(SKILLS_PATH))
    bundles = load_bundles(BUNDLES_PATH)

    rng = random.Random(SEED)

    OUT_PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle_ids = list(bundles.keys())
    if not bundle_ids:
        raise ValueError("No bundles loaded.")

    plan_id = 0
    with OUT_PLANS_PATH.open("w", encoding="utf-8") as f:
        for _ in range(TOTAL_PLANS):
            plan_id += 1
            bid = rng.choice(bundle_ids)
            spec = bundles[bid]
            plan = sample_plan(bid, spec, global_skills, rng)
            plan["plan_id"] = plan_id
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")

    print(f"✅ wrote {plan_id} plans -> {OUT_PLANS_PATH}")
    print(f"bundles={len(bundles)} | total_plans={TOTAL_PLANS} | seed={SEED}")


if __name__ == "__main__":
    run()
