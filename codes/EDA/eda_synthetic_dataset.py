import argparse
import json
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


# --------- Repo root finder (root that contains "codes") ---------
def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start.parent] + list(start.parents):
        if (p / "codes").exists():
            return p
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()

# --------- Paths based on your repo structure ---------
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "synthetic_dataset_expanded.jsonl"  # עדכני אם צריך
SKILLS_PATH = REPO_ROOT / "data" / "skills_v1.txt"
BUNDLES_PATH = REPO_ROOT / "data" / "bundles_v1.json"

OUTPUT_DIR = REPO_ROOT / "EDA outputs"  # מחוץ ל-codes, בתיקיית הפרויקט


# --------- Helpers ---------
def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {i} in {path}: {e}") from e
    return items


def load_skills_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"skills file not found: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_bundles(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"bundles file not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    bundles = obj.get("bundles", [])
    by_id = {b.get("id"): b for b in bundles if b.get("id")}
    return bundles, by_id


def normalize_text(s: str) -> str:
    return (s or "").lower()


def safe_filename(name: str) -> str:
    # keep it filesystem-friendly (but still readable)
    name = name.strip().lower()
    name = re.sub(r"[^\w\s\-]+", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120] if len(name) > 120 else name


def save_fig(fig_title: str, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# --------- FIXED: robust matching (no substring false positives) ---------
def appears_in_text(skill: str, text: str) -> bool:
    """
    True only if the skill appears as a whole token/phrase (not as a substring).
    Prevents false positives like:
      - 'Go' in 'going'
      - 'C' in 'CI/CD'
    Handles punctuation skills too (node.js, c++, c#).
    """
    s = (skill or "").strip().lower()
    t = (text or "").lower()

    if not s:
        return False

    # If skill contains punctuation, match exact string with non-alphanumeric boundaries
    # Example: node.js / c++ / c#
    if re.search(r"[^a-z0-9\s]", s):
        pattern = r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])"
        return re.search(pattern, t) is not None

    # Normalize to tokens (turn punctuation into spaces) for safe word/phrase match
    t_norm = re.sub(r"[^a-z0-9]+", " ", t)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()

    s_norm = re.sub(r"[^a-z0-9]+", " ", s)
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    if not s_norm:
        return False

    pattern = r"(?:^|\s)" + re.escape(s_norm) + r"(?:$|\s)"
    return re.search(pattern, t_norm) is not None


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to expanded dataset jsonl")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    items = read_jsonl(data_path)
    all_skills = load_skills_list(SKILLS_PATH)
    _, bundles_by_id = load_bundles(BUNDLES_PATH)

    # ---------- Counters ----------
    explicit_occ = 0
    implicit_occ = 0

    bundle_counts = Counter()
    role_counts = Counter()

    skill_total = Counter()
    skill_explicit = Counter()
    skill_implicit = Counter()

    # quality / leakage
    implicit_leak_rows = 0
    explicit_missing_rows = 0
    inconsistent_rows = 0

    for ex in items:
        jd = ex.get("job_description", "") or ""
        jd_l = normalize_text(jd)

        bundle = ex.get("bundle")
        if bundle:
            bundle_counts[bundle] += 1
            role = bundles_by_id.get(bundle, {}).get("role")
            if role:
                role_counts[role] += 1

        exp_list = ex.get("explicit_skills", []) or []
        imp_list = ex.get("implicit_skills", []) or []
        skills_dict = ex.get("skills", {}) or {}

        # explicit vs implicit occurrences (count of skill mentions per sample)
        explicit_occ += len(exp_list)
        implicit_occ += len(imp_list)

        # coverage counters
        for s in skills_dict.keys():
            skill_total[s] += 1
        for s in exp_list:
            skill_explicit[s] += 1
        for s in imp_list:
            skill_implicit[s] += 1

        # consistency: skills keys should equal union(explicit+implicit)
        set_union = set(exp_list) | set(imp_list)
        if set(skills_dict.keys()) != set_union:
            inconsistent_rows += 1

        # leakage: any implicit appears in text (FIXED)
        leaked = any(appears_in_text(s, jd) for s in imp_list)
        if leaked:
            implicit_leak_rows += 1

        # explicit missing: any explicit NOT appearing in text (FIXED)
        missing = any(not appears_in_text(s, jd) for s in exp_list)
        if missing:
            explicit_missing_rows += 1

    n = len(items)
    print(f"\nLoaded {n} rows from {data_path}")
    print(f"skills file:  {SKILLS_PATH}")
    print(f"bundles file: {BUNDLES_PATH}")
    print(f"output dir:   {OUTPUT_DIR}\n")

    # ---------- 1) Pie: Explicit vs Implicit occurrences ----------
    plt.figure()
    plt.pie([explicit_occ, implicit_occ], labels=["Explicit (1.0)", "Implicit (0.5)"], autopct="%1.1f%%")
    plt.title("Skill Occurrences: Explicit vs Implicit")
    save_fig(
        fig_title="Skill Occurrences: Explicit vs Implicit",
        filename="01_skill_occurrences_explicit_vs_implicit_pie.png",
    )

    # ---------- 2) Bar: Samples per bundle ----------
    if bundle_counts:
        labels, vals = zip(*bundle_counts.most_common())
        plt.figure(figsize=(max(8, len(labels) * 0.45), 5))
        plt.bar(labels, vals)
        plt.xticks(rotation=60, ha="right")
        plt.title("Samples per Bundle")
        save_fig(
            fig_title="Samples per Bundle",
            filename="02_samples_per_bundle_bar.png",
        )

    # ---------- 2b) Bar: Samples per role (from bundles config) ----------
    if role_counts:
        labels, vals = zip(*role_counts.most_common())
        plt.figure(figsize=(max(8, len(labels) * 0.45), 5))
        plt.bar(labels, vals)
        plt.xticks(rotation=45, ha="right")
        plt.title("Samples per Role (from bundles config)")
        save_fig(
            fig_title="Samples per Role",
            filename="03_samples_per_role_bar.png",
        )

    # ---------- 3) Skill coverage summary ----------
    present = set(skill_total.keys())
    all_set = set(all_skills)

    missing_skills = sorted(all_set - present)
    covered_pct = 100.0 * (len(all_set & present) / max(1, len(all_set)))

    # coverage bar (covered vs missing)
    plt.figure()
    plt.bar(["Covered skills", "Missing skills"], [len(all_set & present), len(missing_skills)])
    plt.title(f"Skill Coverage (overall) — {covered_pct:.2f}%")
    save_fig(
        fig_title="Skill Coverage (overall)",
        filename="04_skill_coverage_overall_bar.png",
    )

    # top skills (overall / explicit / implicit) as bar charts
    def topk_bar(counter: Counter, title: str, filename: str, k: int = 20):
        top = counter.most_common(k)
        if not top:
            return
        skills, counts = zip(*top)
        plt.figure(figsize=(10, 6))
        plt.barh(list(reversed(skills)), list(reversed(counts)))
        plt.title(title)
        save_fig(title, filename)

    topk_bar(skill_total, "Top 20 Skills (Overall Frequency)", "05_top20_skills_overall_barh.png", k=20)
    topk_bar(skill_explicit, "Top 20 Skills (Explicit Frequency)", "06_top20_skills_explicit_barh.png", k=20)
    topk_bar(skill_implicit, "Top 20 Skills (Implicit Frequency)", "07_top20_skills_implicit_barh.png", k=20)

    # ---------- 4) Quality / consistency ----------
    print("=== Coverage ===")
    print(f"Total skills in skills_v1.txt: {len(all_set)}")
    print(f"Skills seen in dataset:        {len(present)}")
    print(f"Coverage:                      {covered_pct:.2f}%")

    print("\n=== Quality / Consistency ===")
    print(f"Inconsistent rows (skills != explicit+implicit): {inconsistent_rows} ({100*inconsistent_rows/max(1,n):.2f}%)")
    print(f"Implicit leakage rows (implicit appears in text): {implicit_leak_rows} ({100*implicit_leak_rows/max(1,n):.2f}%)")
    print(f"Explicit-missing rows (explicit not in text):     {explicit_missing_rows} ({100*explicit_missing_rows/max(1,n):.2f}%)")

    # save quality as a simple bar chart too
    plt.figure()
    plt.bar(
        ["Inconsistent rows", "Implicit leakage rows", "Explicit-missing rows"],
        [inconsistent_rows, implicit_leak_rows, explicit_missing_rows],
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Data Quality Checks (Row Counts)")
    save_fig(
        fig_title="Data Quality Checks (Row Counts)",
        filename="08_data_quality_checks_bar.png",
    )

    # optionally dump missing skills list (text file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    miss_path = OUTPUT_DIR / "missing_skills_never_seen.txt"
    miss_path.write_text("\n".join(missing_skills), encoding="utf-8")
    print(f"\n[SAVED] {miss_path}")


if __name__ == "__main__":
    main()
