# ESG Commitment Verification

**English** | [繁體中文](README.zh-TW.md)

Code and experimental results for **Evaluating Generalization and Company-Level Reliability in ESG Commitment Verification** by TEAM_9924.

This project evaluates Qwen2.5-7B-Instruct on ESG commitment verification across seen companies, unseen companies and unseen industries. It also compares company-level indicator estimation and ranking using ECR, CER and BHCR.

The repository includes 12 experiments, five saved folds per experiment, paragraph predictions, company-level outputs and detailed results. The labeled dataset contains 2,000 Traditional Chinese paragraphs from 49 companies across 18 industries. Model weights and trained adapters must be obtained or trained separately.

## Methods

| Method | Seen company | Unseen company | Unseen industry |
|---|---|---|---|
| QLoRA | `random_qlora` | `company_qlora` | `industry_qlora` |
| Contrastive QLoRA | `random_contrastive` | `company_contrastive` | `industry_contrastive` |
| Frozen Qwen embeddings + Ridge | `random_ridge` | `company_ridge` | `industry_ridge` |
| Frozen Qwen embeddings + pairwise ranking | `random_pairwise` | `company_pairwise` | `industry_pairwise` |

The four paragraph-level tasks are `promise_status`, `verification_timeline`, `evidence_status` and `evidence_quality`. Company indicators summarize evidence coverage (ECR), evidence clarity (CER) and commitment-horizon balance (BHCR).

## Project structure

```text
configs/       Experiment definitions and default arguments
data/          Labeled paragraphs and company/industry metadata
src/           Experiment programs
results/       Saved outputs grouped by split and method
supplement/    Detailed metrics, distributions and fold memberships
scripts/       Experiment launcher and analysis utilities
docs/          Data formats and reproduction details
runs/          New experiment outputs, created when running experiments
```

Each experiment in `results/` contains `fold_0` through `fold_4`, with saved training/test records, split information and method-specific outputs.

## Installation

Use Python 3.10 or newer and a CUDA-capable GPU for model training and inference. Install a PyTorch build compatible with your CUDA environment, then install the project dependencies.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

## Run experiments

List the available experiments:

```bash
python scripts/run_experiment.py --list
```

Run an experiment using the default `Qwen/Qwen2.5-7B-Instruct` base model:

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --output-dir runs/industry_qlora
python scripts/run_experiment.py random_ridge --gpu 0 --output-dir runs/random_ridge
python scripts/run_experiment.py company_pairwise --gpu 0 --output-dir runs/company_pairwise
```

Prepare train/test folds without training:

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --prepare-only --output-dir runs/industry_splits
```

Use a local model directory:

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --output-dir runs/industry_qlora_local --base-model models/Qwen2.5-7B-Instruct
python scripts/run_experiment.py random_ridge --gpu 0 --output-dir runs/random_ridge_local --embedding-model models/Qwen2.5-7B-Instruct
```

Run these commands from the project root. The launcher locates packaged data and code relative to the repository and supplies company/industry metadata when required. Use a new, empty output directory for each run. Additional options pass through to the experiment program; see [experiment definitions](configs/experiments.json).

## Default settings

| Setting | Default |
|---|---|
| Cross-validation | 5 folds, seed 42 |
| QLoRA training | 3 epochs, learning rate 1e-4 |
| QLoRA input length | 4096 tokens |
| Batch size / gradient accumulation | 1 / 4 |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Contrastive objective | lambda=0.10, temperature=0.07 |
| Embedding input length | 1024 tokens |
| Ridge alpha | 10 |

## Results and data

- [Detailed results](supplement/RESULTS.md)
- [Fold-level metrics](supplement/fold_metrics.csv)
- [Means, standard deviations and descriptive intervals](supplement/mean_sd_descriptive_intervals.csv)
- [Fold memberships](supplement/split_membership.csv)
- [Company distribution](supplement/company_distribution.csv) and [industry distribution](supplement/industry_distribution.csv)
- [Data formats and indicator formulas](docs/DATA_AND_FORMATS.md)
- [Reproduction details](docs/REPRODUCIBILITY.md)
