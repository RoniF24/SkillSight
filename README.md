# SkillSight — Explicit vs Implicit Skill Extraction (NLP)

SkillSight is an academic NLP project for detecting **skills** in free-text descriptions and labeling each detected skill as:

- `0.5` = **IMPLICIT** (supported by context, not explicitly written)
- `1.0` = **EXPLICIT** (appears explicitly in the text)

> Note: outputs are **focused** — the `skills` object contains **only detected skills** (non-zero).  
> Non-detected skills are omitted.

---

## Links
- GitHub: https://github.com/RoniF24/SkillSight

### Trained models (Hugging Face)
- Pairwise: https://huggingface.co/Roni1999/pairwise_seed42_epoch3  
- One-Pass (baseline, no class weights): https://huggingface.co/Roni1999/seed43_ep3_baseline  
- One-Pass (with class weights): https://huggingface.co/Roni1999/seed43_ep3_cw  

---

## Project motivation
Keyword-based screening can miss qualified candidates when important skills are **implied** rather than explicitly written.  
This project focuses on detecting **implicit skills** from context and providing **clear labeling** (Implicit vs Explicit).

---

## Problem statement
Given a text (job description / experience paragraph), predict a focused skill profile:
1) which skills are present, and  
2) whether each detected skill is **Explicit** or **Implicit**.

---

## Visual abstract
Figures and the visual abstract are stored under:
- `visuals/`

---

## Datasets used or collected
All datasets and metadata are stored under `data/`, including:
- **Skill list** (global vocabulary)
- **Bundles list** (skill bundles used by the sampler)
- **Synthetic dataset** (two versions: base + extra)

Each dataset example is stored as a JSON object (JSONL format), e.g.:
```json
{
  "job_description": "Built REST endpoints and authentication for a web service using Python.",
  "skills": {
    "Python": 1.0,
    "REST API Design": 0.5
  }
}
```

`splits_v1/` contains a fixed **train/val/test** split and a snapshot of the skill list used for model runs.

---

## Data augmentation and generation methods
We generate a synthetic labeled dataset using an LLM-based pipeline:
1) **Sampling / Planning** (`codes/sampler/`): choose skills (and bundles), desired number of skills, and label each as explicit/implicit.
2) **Text generation** (`codes/generator/`): send prompt templates to an LLM to produce natural job-description text matching the plan.
3) **Validation** (utilities under `codes/config/` and related scripts): validate plans / bundles consistency and enforce labeling rules.

Prompt templates for both dataset generation and the zero-shot baseline are stored under `codes/prompts/`.

---

## Input / Output examples
**Input:** a free-text paragraph (`job_description`).  
**Output:** a focused mapping of detected skills to `{0.5, 1.0}`.

Example output:
```json
{
  "job_description": "...",
  "skills": {
    "Docker": 0.5,
    "Kubernetes": 1.0
  }
}
```

---

## Models and pipelines used
We train and evaluate two approaches:

1) **One-Pass** (text → all skills at once)  
2) **Pairwise** ((text, skill) → NONE / IMPLICIT / EXPLICIT)

Post-processing selects **Top-K skills per example** (typically K in [3..6]) and assigns implicit/explicit based on the predicted class.

---

## Training process and parameters
Training and evaluation scripts are under `codes/models/`.

Typical flow:
1) Prepare train/val/test splits (`codes/models/split_data_for_models/`)
2) Train models (One-Pass / Pairwise)
3) Evaluate and save metrics + error analysis under Results

---

## Metrics
We report:
- Precision@K, Recall@K, F1@K (skill detection)
- Typed metrics (correct only if both skill and label match: IMPLICIT/EXPLICIT)
- Type accuracy on intersection (implicit vs explicit correctness given the skill was detected)

---

## Results
Experiment outputs (JSON/CSV metrics, per-skill reports, error tables, and run folders) are stored under:
- `results of ZS and models/`

Visualizations (EDA/baseline/model plots) are stored under:
- `visuals/`

---

## Repository structure
- `codes/` — project code
  - `baselines/` — zero-shot baseline
  - `config/` — bundle validation utilities
  - `EDA/` — scripts for running EDA on the dataset
  - `generator/` — dataset generation code (LLM pipeline)
  - `models/`
    - `compare_results/` — comparing model results
    - `OnePass/` — One-Pass model code (data loading, training, inference, metrics)
    - `pairwise/` — Pairwise model code (data loading, training, inference, metrics)
    - `split_data_for_models/` — train/val/test split preparation utilities
  - `prompts/` — prompt templates (dataset generation + zero-shot)
  - `sampler/` — plan creation before dataset generation (skills/bundles selection + labeling)
- `data/` — datasets and metadata (skills, bundles, synthetic datasets, `splits_v1/`)
- `trained_onepass/` — saved One-Pass checkpoints/artifacts
- `trained_pairwise/` — saved Pairwise checkpoints/artifacts
- `results of ZS and models/` — baseline + model results (CSV/JSON)
- `slides/` — presentations (PPT/PDF)
- `visuals/` — figures, plots, and visual abstract
- `setup_env.bat` — Windows environment setup (+ optional model download)

---

## How to run (Windows)

### 1) Environment setup
```bat
setup_env.bat
```

### 2) Optional: download trained models from Hugging Face
```bat
setup_env.bat --download-models
```

### 3) Run examples (main entry points)
Use `--help` on each script to see all options.

**A) Zero-shot baseline**
```bash
python codes/baselines/pure_zero_shot.py --help
```

**B) Train One-Pass**
```bash
python codes/models/OnePass/train_onepass.py --help
```

**C) Train Pairwise**
```bash
python codes/models/pairwise/train_pairwise.py --help
```

**D) Evaluate / compare results**
```bash
python codes/models/compare_results/compare_models.py --help
```

Outputs are typically written under:
- `results of ZS and models/`

---

## Team members
- Yonatal Elman  
- Michael Kovalchuk  
- Roni Fadlon  
