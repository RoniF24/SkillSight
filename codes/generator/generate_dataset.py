"""codes/generator/generate_dataset.py

Synthetic dataset generator for SkillSight.

This script generates short, resume-style job paragraphs/snippets based on either:
- Plans mode (preferred): reads structured plans from a JSONL file and generates one sample per plan.
- Bundles mode (fallback): samples skill bundles and tries to hit a target implicit/explicit ratio.

By default, this script runs using the Ollama backend (local model). In our experiments, we also generated data
using OpenAI's `gpt-4o-mini` to get stronger, higher-quality outputs.

Backends:
- ollama: local LLMs served by Ollama.
- openai: OpenAI chat models.

Examples
--------

OpenAI (gpt-4o-mini) - the command we used in the 4o-mini run:
    python codes/generator/generate_dataset.py --backend openai --model gpt-4o-mini --n 10 --temperature 0

Note: for the OpenAI backend you must export/set `OPENAI_API_KEY` in your environment.

Ollama (default backend):
    python codes/generator/generate_dataset.py --n 10 --temperature 0

Ollama (explicit backend + model example):
    python codes/generator/generate_dataset.py --backend ollama --model llama3.1:8b --n 10 --temperature 0
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set


# -------------------------
# Repo-root path handling
# -------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2] if len(THIS_FILE.parents) >= 3 else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------------
# Imports (local project)
# -------------------------

try:
    from codes.generator.implicit_hints import (
        IMPLICIT_ANCHORS_SOFT,
        IMPLICIT_ANCHORS_STRONG,
        LEAK_ALIASES,
        FORBIDDEN_TERMS,
    )
except Exception as e:
    raise RuntimeError(
        "Failed to import codes.generator.implicit_hints. "
        "Make sure you run from repo root and that codes/ is importable."
    ) from e


# -------------------------
# Lightweight CLI parsing
# -------------------------

@dataclass
class Args:
    n: int = 1000
    max_retries: int = 24
    strong_after: int = 8
    strong_anchors_per_skill: int = 2

    backend: str = "ollama"  # "ollama" | "openai"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    seed: Optional[int] = None

    # Percent-metric controls (to keep outputs varied and avoid repetitive "30%"-style metrics).
    percent_prob: float = 0.35       # default: ~1/3 of samples may include a % metric (unless a plan overrides)
    recent_percent_window: int = 15  # avoid repeating % values from the last N accepted samples

    # Plans mode (preferred)
    plans_path: str = "data/plans/plans_v1.jsonl"
    shuffle_plans: bool = False  # default: keep order!
    show_text: bool = False      # print generated paragraph/snippet live

    # Fallback bundles mode
    k: int = 6
    implicit_ratio_target: float = 0.667
    bundles_path: str = "data/bundles_v1.json"

    # Prompt template
    master_prompt_path: str = "codes/prompts/master_prompt.txt"

    out_dir: str = "outputs"
    append: bool = False
    verbose: bool = True


def _parse_bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_args(argv: List[str]) -> Args:
    try:
        import argparse as _argparse  # noqa
        if not hasattr(_argparse, "ArgumentParser"):
            raise ImportError("argparse module missing ArgumentParser")
    except Exception:
        _argparse = None  # type: ignore

    a = Args()

    if _argparse is None:
        # Fallback minimal parser.
        # Supports both `--key=value` and `--key value` forms to keep the script runnable even in constrained
        # environments where `argparse` is unavailable.
        i = 0
        while i < len(argv):
            tok = argv[i]
            if not tok.startswith("--"):
                i += 1
                continue

            key: str
            val: Optional[str]

            if "=" in tok:
                key, val = tok[2:].split("=", 1)
            else:
                key = tok[2:]
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    val = argv[i + 1]
                    i += 1
                else:
                    val = "true"

            key = key.replace("-", "_")
            if hasattr(a, key):
                cur = getattr(a, key)
                if isinstance(cur, bool):
                    setattr(a, key, _parse_bool(val or "false"))
                elif isinstance(cur, int):
                    setattr(a, key, int(val))  # type: ignore
                elif isinstance(cur, float):
                    setattr(a, key, float(val))  # type: ignore
                else:
                    setattr(a, key, val)  # type: ignore

            i += 1
        return a

    ap = _argparse.ArgumentParser(
        description="Generate synthetic job paragraphs from prepared plans (ordered) or bundles, with anchor-based implicit cues."
    )
    ap.add_argument("--n", type=int, default=a.n)
    ap.add_argument("--max-retries", type=int, default=a.max_retries)
    ap.add_argument("--strong-after", type=int, default=a.strong_after)
    ap.add_argument("--strong-anchors-per-skill", type=int, default=a.strong_anchors_per_skill)

    ap.add_argument("--backend", type=str, default=a.backend, choices=["ollama", "openai"])
    ap.add_argument("--model", type=str, default=a.model)
    ap.add_argument("--temperature", type=float, default=a.temperature)
    ap.add_argument("--seed", type=int, default=None)

    # Percent controls
    ap.add_argument("--percent-prob", type=float, default=a.percent_prob)
    ap.add_argument("--recent-percent-window", type=int, default=a.recent_percent_window)

    ap.add_argument("--plans-path", type=str, default=a.plans_path)
    ap.add_argument("--shuffle-plans", action="store_true", default=a.shuffle_plans)
    ap.add_argument("--show-text", action="store_true", default=a.show_text)

    ap.add_argument("--k", type=int, default=a.k)
    ap.add_argument("--implicit-ratio-target", type=float, default=a.implicit_ratio_target)
    ap.add_argument("--bundles-path", type=str, default=a.bundles_path)

    ap.add_argument("--master-prompt-path", type=str, default=a.master_prompt_path)
    ap.add_argument("--out-dir", type=str, default=a.out_dir)
    ap.add_argument("--append", action="store_true", default=a.append)
    ap.add_argument("--verbose", action="store_true", default=a.verbose)

    ns = ap.parse_args(argv)

    return Args(
        n=ns.n,
        max_retries=ns.max_retries,
        strong_after=ns.strong_after,
        strong_anchors_per_skill=ns.strong_anchors_per_skill,
        backend=ns.backend,
        model=ns.model,
        temperature=ns.temperature,
        seed=ns.seed,
        percent_prob=ns.percent_prob,
        recent_percent_window=ns.recent_percent_window,
        plans_path=ns.plans_path,
        shuffle_plans=ns.shuffle_plans,
        show_text=ns.show_text,
        k=ns.k,
        implicit_ratio_target=ns.implicit_ratio_target,
        bundles_path=ns.bundles_path,
        master_prompt_path=ns.master_prompt_path,
        out_dir=ns.out_dir,
        append=ns.append,
        verbose=ns.verbose,
    )


# -------------------------
# Prompt helpers
# -------------------------

DOMAINS = [
    "HealthTech", "FinTech", "E-commerce", "Logistics", "EdTech", "Telecom",
    "Gaming", "SaaS", "Cybersecurity", "Data Platform"
]
SENIORITIES = ["Junior", "Mid-level", "Senior", "Staff"]
ROLE_TITLES = [
    "Backend Engineer", "Platform Engineer", "Site Reliability Engineer",
    "Security Engineer", "Data Engineer", "Software Engineer", "QA Engineer"
]
TONES = ["clear", "technical", "impact-oriented", "concise", "pragmatic"]

OPENERS = [
    "In my previous role",
    "In my current position",
    "Earlier in my career",
    "As a core member of the engineering team",
    "While working on our main product",
    "As part of the platform team",
    "While maintaining our production systems",
    "During a large-scale migration",
    "When we prepared for a major release",
    "As part of an incident response effort",
    "While improving our deployment pipeline",
    "As part of an ongoing reliability initiative",
    "While scaling the system to handle increased traffic",
    "On a project to modernize our stack",
    "During my day-to-day work on the backend",
    "On the team responsible for our core services",
    "As part of the reliability and performance efforts",
]

BANNED_META_PATTERNS = [
    r"\bparagraph\b",
    r"\bjob description\b",
    r"\bthe following\b",
    # Use a unicode escape for the curly apostrophe to avoid non-ASCII characters in the source.
    r"\bhere(?:'|\u2019)s\b",
    r"\bAs an AI\b",
]

BANNED_BOILERPLATE_PHRASES = [
    "worked closely with cross-functional teams",
    "strong communication skills",
    "team player",
    "fast-paced environment",
]

# ---- Metrics anti-repetition policy ----
# The generator tries to reduce repetitive "nice round" numbers that models tend to overuse.
# We keep 40 banned by default because it shows up disproportionately often; expand if needed.
BANNED_PERCENT_NUMBERS = {40}

# A curated pool to steer the model away from always choosing 30/25/40.
# We do NOT include 25/30/40 here on purpose.
PCT_POOL = [
    7, 9, 11, 13, 14, 16, 17, 19, 21, 22, 23, 27, 28, 31, 33, 34, 37, 41, 43, 47, 52
]

METRIC_TOPIC_TEMPLATES = [
    "deployment time",
    "API latency",
    "query response time",
    "incident rate",
    "error rate",
    "on-call alerts",
    "timeout rate",
    "support tickets",
]

# regexes for catching percent usage
PERCENT_TOKEN_RE = re.compile(r"\b(\d{1,3})\s*%\b", flags=re.IGNORECASE)
PERCENT_WORD_RE = re.compile(r"\b(\d{1,3})\s*percent\b", flags=re.IGNORECASE)
PERCENT_PHRASE_OVER_RE = re.compile(r"\bover\s+(\d{1,3})\s*%\b", flags=re.IGNORECASE)


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def build_flexible_phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\/", r"(?:\/|\s+\/\s+)")
    escaped = escaped.replace(r"\-", r"(?:\-|\s+)")
    escaped = escaped.replace(r"\_", r"(?:\_|\s+)")
    return re.compile(escaped, flags=re.IGNORECASE)


def extract_percent_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for m in PERCENT_TOKEN_RE.finditer(text):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            pass
    for m in PERCENT_WORD_RE.finditer(text):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            pass
    for m in PERCENT_PHRASE_OVER_RE.finditer(text):
        try:
            nums.append(int(m.group(1)))
        except Exception:
            pass
    # unique, preserve order
    seen: Set[int] = set()
    out: List[int] = []
    for n in nums:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def choose_percent(avoid_percent_numbers: Set[int]) -> int:
    """
    Choose a percent that is NOT:
      - in avoid_percent_numbers (recent window)
      - in BANNED_PERCENT_NUMBERS
    Falls back gracefully.
    """
    candidates = [p for p in PCT_POOL if p not in avoid_percent_numbers and p not in BANNED_PERCENT_NUMBERS]
    if candidates:
        return random.choice(candidates)

    candidates = [p for p in PCT_POOL if p not in BANNED_PERCENT_NUMBERS]
    if candidates:
        return random.choice(candidates)

    # last resort
    return random.choice(PCT_POOL) if PCT_POOL else 17


def explicit_present(text: str, skill: str) -> bool:
    """
    Robust explicit matching:
    - Handles C#, C++ where \b breaks on #/+
    - Special-cases single-letter skills like "C" so we don't accept "C++"/"C#"
    - Handles *.js variants ("Node.js" vs "Node js")
    """
    t = text
    s = skill.strip()

    if not s:
        return False

    if s.endswith(".js"):
        base = s[:-3]
        pat = re.compile(rf"(?<!\w){re.escape(base)}(?:\.|\s)?js(?!\w)", flags=re.IGNORECASE)
        return bool(pat.search(t))

    # Single-letter skills (e.g., "C"):
    # Accept: "C", "C,", "C.", "C11", "C99", "C language"
    # Reject: "C++", "C#"
    if len(s) == 1:
        tok = re.escape(s)
        pat = re.compile(rf"(?<![A-Za-z0-9_]){tok}(?![A-Za-z_+#])", flags=re.IGNORECASE)
        return bool(pat.search(t))

    # single token (includes C++, C#, etc.)
    if " " not in s:
        tok = re.escape(s)
        pat = re.compile(rf"(?<!\w){tok}(?!\w)", flags=re.IGNORECASE)
        return bool(pat.search(t))

    # multi token
    tokens = [re.escape(tok) for tok in s.split()]
    pat = re.compile(r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)", flags=re.IGNORECASE)
    return bool(pat.search(t))


def contains_any_forbidden(text: str, terms: List[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        term_l = term.lower().strip()
        if not term_l:
            continue
        if re.search(rf"(?<!\w){re.escape(term_l)}(?!\w)", lowered):
            return True
    return False


def meta_text_violation(text: str) -> Optional[str]:
    for pat in BANNED_META_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return pat
    return None


def boilerplate_violation(text: str) -> Optional[str]:
    low = text.lower()
    for phrase in BANNED_BOILERPLATE_PHRASES:
        if phrase in low:
            return phrase
    return None


def metric_violation(
    text: str,
    allow_percent: bool,
    avoid_percent_numbers: Set[int],
    required_percent: Optional[int],
) -> Optional[str]:
    """
    If allow_percent=False -> forbid ANY % usage.
    If allow_percent=True -> forbid:
      - banned fixed numbers (e.g., 40)
      - numbers that appeared recently (avoid_percent_numbers)
      - if any % appears AND required_percent is set: require that % number (forces randomness)
    """
    nums = extract_percent_numbers(text)

    if not allow_percent and nums:
        return "percent_not_allowed"

    if allow_percent and nums:
        for n in nums:
            if n in BANNED_PERCENT_NUMBERS:
                return f"percent_banned:{n}"
            if n in avoid_percent_numbers:
                return f"percent_repeated:{n}"

        if required_percent is not None and required_percent not in nums:
            return f"percent_wrong:want_{required_percent}"

    return None


# -------------------------
# Plans loading (preferred)
# -------------------------

def load_plans_jsonl(path: Path) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Plan line {line_no} is not a JSON object in {path}")

            for key in ("explicit", "implicit", "domain", "seniority", "plan_id"):
                if key not in obj:
                    raise ValueError(f"Missing key '{key}' on line {line_no} in {path}")

            if not isinstance(obj.get("explicit"), list) or not isinstance(obj.get("implicit"), list):
                raise ValueError(f"'explicit'/'implicit' must be lists on line {line_no} in {path}")

            plans.append(obj)

    if not plans:
        raise ValueError(f"No plans loaded from: {path}")
    return plans


# -------------------------
# Bundle sampling (fallback)
# -------------------------

def load_bundles(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "bundles" not in obj or not isinstance(obj["bundles"], list):
        raise ValueError("bundles_v1.json must contain {'bundles': [...]} at top-level.")
    return obj["bundles"]


def sample_plan(bundle: Dict[str, Any], k: int, implicit_ratio_target: float, plan_id: int) -> Dict[str, Any]:
    must_have: List[str] = list(bundle.get("must_have", []))
    optional: List[str] = list(bundle.get("optional", []))
    pick_one_of: List[List[str]] = list(bundle.get("pick_one_of", []))

    explicit = must_have[:]
    implicit: List[str] = []

    for group in pick_one_of:
        if group:
            implicit.append(random.choice(group))

    remaining = max(0, k - len(explicit))
    needed = max(0, remaining - len(implicit))
    if needed > 0:
        pool = [x for x in optional if x not in implicit and x not in explicit]
        if len(pool) < needed:
            needed = len(pool)
        implicit.extend(random.sample(pool, k=needed))

    all_skills = explicit + implicit

    domain = random.choice(DOMAINS)
    seniority = random.choice(SENIORITIES)
    role_title = random.choice(ROLE_TITLES)

    return {
        "bundle": bundle.get("id", "unknown_bundle"),
        "k": len(all_skills),
        "implicit_ratio_target": implicit_ratio_target,
        "explicit": explicit,
        "implicit": implicit,
        "all_skills": all_skills,
        "domain": domain,
        "seniority": seniority,
        "role_title": role_title,
        "plan_id": plan_id,
    }


# -------------------------
# Anchor selection
# -------------------------

def pick_anchors_for_skill(skill: str, strong: bool, n: int) -> List[str]:
    bank = IMPLICIT_ANCHORS_STRONG if strong else IMPLICIT_ANCHORS_SOFT
    phrases = bank.get(skill, []) or []
    phrases = list(dict.fromkeys([p.strip() for p in phrases if isinstance(p, str) and p.strip()]))

    if not phrases:
        return []
    if n <= 1:
        return [random.choice(phrases)]
    if len(phrases) >= n:
        return random.sample(phrases, k=n)
    return phrases[:]  # not enough -> return all uniques (no duplicates)


def build_implicit_requirements(
    implicit_skills: List[str],
    strong: bool,
    anchors_per_skill: int,
    use_anchors: bool = True,
) -> Tuple[str, Dict[str, List[str]]]:
    selected: Dict[str, List[str]] = {}
    if not implicit_skills:
        return "", selected

    if not use_anchors:
        lines: List[str] = [
            "For each of the following implicit skills, describe realistic usage of this skill in the story,",
            "without naming the skill itself explicitly. Make the connection clear but natural:",
        ]
        for s in implicit_skills:
            lines.append(
                f"- {s}: weave in concrete actions, tools, or workflows that strongly suggest this skill,"
                f" but do NOT write the name '{s}' anywhere."
            )
        return "\n".join(lines), selected

    lines: List[str] = []
    for s in implicit_skills:
        want = anchors_per_skill if strong else 1
        anchors = pick_anchors_for_skill(s, strong=strong, n=want)
        selected[s] = anchors

        if not anchors:
            lines.append(f"- {s}: (no anchors configured!) add anchors in implicit_hints.py")
            continue

        if strong and len(anchors) >= anchors_per_skill:
            joined = "', '".join(anchors[:anchors_per_skill])
            lines.append(f"- {s}: include BOTH exact phrases '{joined}' (do NOT mention '{s}')")
        else:
            line = f"- {s}: include the exact phrase '{anchors[0]}' (do NOT mention '{s}')"
            if strong and len(anchors) < anchors_per_skill:
                line += " [NOTE: only 1 strong anchor available; add more in implicit_hints.py]"
            lines.append(line)

    return "\n".join(lines), selected


# -------------------------
# LLM backends
# -------------------------

def call_ollama(prompt: str, model: str, temperature: float) -> str:
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    return obj.get("response", "") or ""


def call_openai(prompt: str, model: str, temperature: float) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = base + "/responses"

    payload = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": temperature,
        "max_output_tokens": 500,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="replace"))

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


def llm_generate(prompt: str, backend: str, model: str, temperature: float) -> str:
    if backend == "ollama":
        return call_ollama(prompt, model=model, temperature=temperature)
    if backend == "openai":
        return call_openai(prompt, model=model, temperature=temperature)
    raise ValueError(f"Unknown backend: {backend}")


# -------------------------
# Validation
# -------------------------

def validate_text(
    text: str,
    explicit_skills: List[str],
    implicit_skills: List[str],
    selected_anchors: Dict[str, List[str]],
    strong: bool,
    anchors_per_skill_strong: int,
    allow_percent: bool,
    avoid_percent_numbers: Set[int],
    required_percent: Optional[int],
) -> Tuple[bool, str, int]:
    t = normalize_ws(text)

    mpat = meta_text_violation(t)
    if mpat:
        return False, f"meta_text_banned:{mpat}", 0

    bphrase = boilerplate_violation(t)
    if bphrase:
        return False, f"boilerplate_banned:{bphrase}", 0

    mviol = metric_violation(
        t,
        allow_percent=allow_percent,
        avoid_percent_numbers=avoid_percent_numbers,
        required_percent=required_percent,
    )
    if mviol:
        return False, f"metric_banned:{mviol}", 0

    # explicit skills must appear verbatim
    for s in explicit_skills:
        if not explicit_present(t, s):
            return False, f"missing_explicit:{s}", 0

    # implicit: no forbidden terms / aliases
    for s in implicit_skills:
        forbid = FORBIDDEN_TERMS.get(s, [])[:]
        forbid += LEAK_ALIASES.get(s, [])
        if contains_any_forbidden(t, forbid):
            return False, f"implicit_leak:{s}", 0

    # Implicit anchors are only enforced when we actually requested them.
    # When `use_anchors=False` in `build_prompt`, `selected_anchors` is empty and we skip this check.
    anchor_hits_total = 0
    if selected_anchors:
        for s in implicit_skills:
            anchors = selected_anchors.get(s, [])
            if not anchors:
                continue

            required = 1
            if strong and len(anchors) >= anchors_per_skill_strong:
                required = anchors_per_skill_strong

            hits = 0
            for a in anchors:
                if build_flexible_phrase_pattern(a).search(t):
                    hits += 1

            anchor_hits_total += hits
            if hits < required:
                return False, f"missing_anchor{required}:{s}", anchor_hits_total

    return True, "ok", anchor_hits_total


# -------------------------
# Prompt building
# -------------------------

def build_prompt(
    master_template: str,
    plan: Dict[str, Any],
    strong: bool,
    anchors_per_skill_strong: int,
    correction: Optional[str],
    previous_text: Optional[str],
    allow_percent: bool,
    avoid_percent_numbers: Set[int],
    required_percent: Optional[int],
) -> Tuple[str, Dict[str, List[str]]]:
    explicit: List[str] = list(plan.get("explicit", []))
    implicit: List[str] = list(plan.get("implicit", []))

    # We don't enforce anchors on every sample: mixing "anchored" and "anchor-free" generation tends to
    # produce more diverse text and makes the generator more robust when anchors are missing for some skills.
    use_anchors = random.random() < 0.7  # ~70% with anchors, ~30% without anchors

    implicit_reqs, selected = build_implicit_requirements(
        implicit_skills=implicit,
        strong=strong,
        anchors_per_skill=anchors_per_skill_strong,
        use_anchors=use_anchors,
    )

    forbidden_words = set()
    for s in implicit:
        forbidden_words |= set(FORBIDDEN_TERMS.get(s, []))
        forbidden_words |= set(LEAK_ALIASES.get(s, []))
    forbidden_words |= set(BANNED_BOILERPLATE_PHRASES)
    forbidden_words |= {"paragraph", "job description"}  # meta

    # If this sample should NOT include percent metrics, forbid them explicitly in the prompt too.
    if not allow_percent:
        forbidden_words |= {"%", "percent", "percentage"}

    forbidden_words_str = ", ".join(sorted({w for w in forbidden_words if isinstance(w, str) and w.strip()}))

    opener = random.choice(OPENERS)
    company = random.choice(
        ["GreenLeaf Solutions", "OrbitPay", "BlueNova Systems", "SkyCart Commerce", "CloudHarbor", "MedPulse Health"]
    )

    domain = plan.get("domain") or random.choice(DOMAINS)
    seniority = plan.get("seniority") or random.choice(SENIORITIES)
    role_title = plan.get("role_title") or random.choice(ROLE_TITLES)

    # Metric guidance injected always (first try too) to reduce "30%" bias.
    metric_hint = ""
    if allow_percent:
        avoid_str = ", ".join(str(x) for x in sorted(avoid_percent_numbers | BANNED_PERCENT_NUMBERS))
        topic = random.choice(METRIC_TOPIC_TEMPLATES)
        if required_percent is not None:
            metric_hint = (
                "METRIC GUIDANCE:\n"
                f"- If you include a % metric, use EXACTLY {required_percent}% and tie it to {topic}.\n"
                f"- Avoid these numbers: {avoid_str if avoid_str else 'none'}.\n"
                "- Or you may omit % entirely and use a qualitative outcome.\n"
            )
        else:
            metric_hint = (
                "METRIC GUIDANCE:\n"
                f"- If you include a % metric, avoid these numbers: {avoid_str if avoid_str else 'none'}.\n"
                "- Or you may omit % entirely and use a qualitative outcome.\n"
            )
    else:
        metric_hint = (
            "METRIC GUIDANCE:\n"
            "- Do NOT use any percent metrics. Use a qualitative outcome or a non-% metric (e.g., fewer incidents, fewer tickets).\n"
        )

    correction_block = ""

    if correction:
        correction_block = (
            "CORRECTION (retry):\n"
            f"- Previous attempt failed because: {correction}\n"
        )

        if correction.startswith("missing_explicit:"):
            missing = correction.split(":", 1)[1].strip()

            if len(missing) == 1:
                correction_block += (
                    f"- IMPORTANT: Include the token '{missing}' as a STANDALONE mention (e.g., write \"in {missing}\", "
                    f"\"{missing} language\", or \"{missing}11\").\n"
                    f"- DO NOT replace '{missing}' with '{missing}++' or '{missing}#'.\n"
                )
            else:
                correction_block += (
                    f"- IMPORTANT: Include the exact phrase '{missing}' verbatim (case-insensitive OK) "
                    "in the FIRST sentence. Do NOT paraphrase it.\n"
                )

            correction_block += "- Include ALL explicit phrases exactly as written (do NOT rewrite them).\n"

        elif correction.startswith("missing_anchor"):
            skill = correction.split(":", 1)[1].strip()
            anchors = selected.get(skill, [])
            if anchors:
                if correction.startswith("missing_anchor2") and len(anchors) >= anchors_per_skill_strong:
                    correction_block += (
                        f"- IMPORTANT: For '{skill}', include BOTH exact anchor phrases: "
                        f"'{anchors[0]}' AND '{anchors[1]}'. Do NOT mention '{skill}'.\n"
                    )
                else:
                    correction_block += (
                        f"- IMPORTANT: For '{skill}', include the exact anchor phrase: "
                        f"'{anchors[0]}'. Do NOT mention '{skill}'.\n"
                    )

        elif correction == "duplicate_text":
            correction_block += "- IMPORTANT: Rewrite with different details/incident/metrics; do not reuse the same phrasing.\n"

        elif correction.startswith("metric_banned:"):
            correction_block += (
                "- IMPORTANT: Fix the metric rule:\n"
                "  - If % is not allowed, remove any % / percent / percentage.\n"
                "  - If % is allowed, use the requested % (and avoid banned/recent numbers).\n"
            )

        correction_block += metric_hint
        correction_block += "- Rewrite the paragraph to fix the issue while keeping it natural.\n"
        if previous_text:
            correction_block += f"- Previous text (for reference): {previous_text}\n"
    else:
        # on first attempt, include only the metric hint (short) via CORRECTION_BLOCK slot
        correction_block = metric_hint.strip()

    prompt = master_template.format(
        OPENER=opener,
        COMPANY=company,
        DOMAIN=domain,
        SENIORITY=seniority,
        ROLE_TITLE=role_title,
        TONE=random.choice(TONES),
        EXPLICIT_MUST_INCLUDE=", ".join(explicit),
        IMPLICIT_REQUIREMENTS=implicit_reqs,
        FORBIDDEN_WORDS=forbidden_words_str,
        CORRECTION_BLOCK=correction_block.strip(),
    )
    return prompt, selected


# -------------------------
# Output
# -------------------------

def make_out_paths(out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_dir = out_dir / "full"
    slim_dir = out_dir / "slim"
    full_dir.mkdir(parents=True, exist_ok=True)
    slim_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"synthetic_dataset_{ts}.jsonl"

    return full_dir / base_name, slim_dir / base_name


# -------------------------
# Main loop
# -------------------------

def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.seed is not None:
        random.seed(args.seed)

    master_prompt_path = Path(args.master_prompt_path)
    if not master_prompt_path.is_absolute():
        master_prompt_path = REPO_ROOT / master_prompt_path

    if args.verbose:
        print(f"[INFO] master_prompt_path -> {master_prompt_path.resolve()}")
        print(f"[INFO] master_prompt_exists -> {master_prompt_path.exists()}")

    master_template = load_text_file(master_prompt_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    full_out_path, slim_out_path = make_out_paths(out_dir)

    mode = "a" if args.append else "w"
    full_f = full_out_path.open(mode, encoding="utf-8")
    slim_f = slim_out_path.open(mode, encoding="utf-8")

    if args.verbose:
        print(f"[INFO] Generating {args.n} examples")
        print(f"[INFO] Full Output  -> {full_out_path} (append={args.append})")
        print(f"[INFO] Slim Output  -> {slim_out_path} (append={args.append})")
        print(f"[INFO] Backend={args.backend} model={args.model} temp={args.temperature}")
        print(f"[INFO] percent_prob={args.percent_prob} recent_percent_window={args.recent_percent_window}")

    plans: Optional[List[Dict[str, Any]]] = None
    bundles: Optional[List[Dict[str, Any]]] = None

    plans_path = Path((args.plans_path or "").strip())
    if str(plans_path).strip():
        if not plans_path.is_absolute():
            plans_path = REPO_ROOT / plans_path
        if plans_path.exists():
            plans = load_plans_jsonl(plans_path)

            plans.sort(key=lambda p: int(p.get("plan_id", 10**9)))
            if args.shuffle_plans:
                random.shuffle(plans)

            if args.verbose:
                print(f"[INFO] plans_path -> {plans_path.resolve()}")
                print(f"[INFO] loaded_plans -> {len(plans)}")
                print(f"[INFO] shuffle_plans -> {args.shuffle_plans}")
        else:
            if args.verbose:
                print(f"[WARN] plans_path not found -> {plans_path.resolve()} (fallback to bundles)")

    if plans is None:
        bundles_path = Path(args.bundles_path)
        if not bundles_path.is_absolute():
            bundles_path = REPO_ROOT / bundles_path
        if args.verbose:
            print(f"[INFO] bundles_path -> {bundles_path.resolve()}")
            print(f"[INFO] bundles_exists -> {bundles_path.exists()}")
        bundles = load_bundles(bundles_path)

    produced = 0
    recent_hashes: List[str] = []
    recent_percent_numbers: List[int] = []  # store numbers used in accepted samples

    plan_stream: List[Dict[str, Any]] = []
    if plans is not None:
        if args.n <= len(plans):
            plan_stream = [dict(p) for p in plans[:args.n]]
        else:
            # If the user requests more samples than there are plans, repeat the plan list to reach `n`.
            reps = (args.n + len(plans) - 1) // len(plans)
            big = (plans * reps)[:args.n]
            plan_stream = [dict(p) for p in big]

    for i in range(args.n):
        if plans is not None:
            plan = plan_stream[i]
        else:
            assert bundles is not None
            plan = sample_plan(
                random.choice(bundles),
                k=args.k,
                implicit_ratio_target=args.implicit_ratio_target,
                plan_id=i + 1,
            )

        # Decide metric rule ONCE per sample (but allow plan override if present)
        plan_allow = plan.get("allow_percent_metric", None)
        if isinstance(plan_allow, bool):
            allow_percent = plan_allow
        else:
            allow_percent = (random.random() < float(args.percent_prob))

        avoid_recent = set(recent_percent_numbers[-int(args.recent_percent_window):])

        # If percent metrics are allowed, choose a target percent and enforce it if any % appears.
        # This is a simple way to encourage variety and avoid the model defaulting to the same numbers.
        required_percent: Optional[int] = None
        if allow_percent:
            required_percent = choose_percent(avoid_recent)

        correction: Optional[str] = None
        prev_text: Optional[str] = None
        ok = False

        for attempt in range(1, args.max_retries + 1):
            strong = attempt >= args.strong_after

            prompt, selected_anchors = build_prompt(
                master_template=master_template,
                plan=plan,
                strong=strong,
                anchors_per_skill_strong=args.strong_anchors_per_skill,
                correction=correction,
                previous_text=prev_text,
                allow_percent=allow_percent,
                avoid_percent_numbers=avoid_recent,
                required_percent=required_percent,
            )

            try:
                gen = llm_generate(prompt, backend=args.backend, model=args.model, temperature=args.temperature)
            except Exception as e:
                print(f"[FAIL] {i+1}/{args.n} plan_id={plan.get('plan_id','?')} reason=backend_error:{e}")
                break

            gen = normalize_ws(gen)

            h = str(hash(gen))
            # Avoid near-duplicate outputs within a run (hash is only used as an in-memory fingerprint).
            if h in recent_hashes:
                correction = "duplicate_text"
                prev_text = gen[:220]
                if args.verbose:
                    print(f"[RETRY] {i+1}/{args.n} plan_id={plan.get('plan_id','?')} try={attempt} reason=duplicate_text")
                continue

            ok, reason, anchor_hits = validate_text(
                gen,
                explicit_skills=list(plan.get("explicit", [])),
                implicit_skills=list(plan.get("implicit", [])),
                selected_anchors=selected_anchors,
                strong=strong,
                anchors_per_skill_strong=args.strong_anchors_per_skill,
                allow_percent=allow_percent,
                avoid_percent_numbers=avoid_recent,
                required_percent=required_percent,
            )

            if ok:
                recent_hashes.append(h)
                if len(recent_hashes) > 25:
                    recent_hashes.pop(0)

                # Update metric memory only for accepted samples
                if allow_percent:
                    nums = extract_percent_numbers(gen)
                    if nums:
                        # if we enforced required_percent, it's usually included; store actual nums anyway
                        recent_percent_numbers.extend(nums)
                        if len(recent_percent_numbers) > 200:
                            recent_percent_numbers = recent_percent_numbers[-200:]

                skills_dict: Dict[str, float] = {}
                for s in plan.get("explicit", []):
                    skills_dict[str(s)] = 1.0
                for s in plan.get("implicit", []):
                    skills_dict[str(s)] = 0.5

                full_row: Dict[str, Any] = {
                    "job_description": gen,
                    "skills": skills_dict,
                    "domain": plan.get("domain"),
                    "seniority": plan.get("seniority"),
                    "bundle": plan.get("bundle"),
                    "plan_id": plan.get("plan_id"),
                    "explicit_skills": list(plan.get("explicit", [])),
                    "implicit_skills": list(plan.get("implicit", [])),
                    "allow_percent_metric": allow_percent,
                }

                slim_row: Dict[str, Any] = {
                    "job_description": gen,
                    "skills": skills_dict,
                }

                full_f.write(json.dumps(full_row, ensure_ascii=False) + "\n")
                slim_f.write(json.dumps(slim_row, ensure_ascii=False) + "\n")
                full_f.flush()
                slim_f.flush()

                produced += 1
                if args.verbose:
                    print(
                        f"[OK] {produced}/{args.n} plan_id={plan.get('plan_id','?')} "
                        f"try={attempt} strong={strong} anchors={anchor_hits} "
                        f"allow_percent={allow_percent} req_pct={required_percent}"
                    )
                if args.show_text:
                    print(f"[TEXT plan_id={plan.get('plan_id','?')}] {gen}\n")
                break

            correction = reason
            prev_text = gen[:240]
            if args.verbose:
                print(f"[RETRY] {i+1}/{args.n} plan_id={plan.get('plan_id','?')} try={attempt} reason={reason}")

        if not ok:
            print(f"[FAIL] {i+1}/{args.n} plan_id={plan.get('plan_id','?')} reason={correction or 'unknown'}")

    full_f.close()
    slim_f.close()
    print(f"[DONE] wrote {produced} rows ->")
    print(f"       full -> {full_out_path}")
    print(f"       slim -> {slim_out_path}")


if __name__ == "__main__":
    main()
