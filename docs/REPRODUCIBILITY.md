# Reproduction and provenance

[English README](../README.md) | [中文 README](../README.zh-TW.md)

## What was preserved and changed

All 644 supplied files were retained, with legacy console paths sanitized for the public release: 12 Python programs, 398 CSVs, 216 JSONs, six compressed embedding caches and 12 text logs. CSV/JSON metrics, splits, predictions and embedding arrays were preserved. Legacy console logs retain their measurements but replace private machine/user output paths with repository-relative paths. The company metadata CSV was moved to `data/`. The 12 Python programs were moved to `src/`; old machine-specific path examples were replaced by repository-relative examples, the industry metadata filename was corrected, and the import-time hard-coded GPU 1 setting was replaced by early selection from `--gpu` (default 0). Training algorithms, prompts, metrics, fold builders and hyperparameter defaults were retained.

`data/vpesg4k_train_val.json` was reconstructed from `results/random/qlora/fold_0/{train,test}.json` using the saved `train_indices` and `test_indices` in `split_info.json`. This recovers the saved input order, rather than guessing by sorting IDs. The audit checks positional indices in all archives that supply them and checks original record fields against all 60 train/test partitions. Industry-enriched records may add fields. The separate original input file and its original byte hash were not supplied; logical records/order have been recovered, not original file serialization.

Each experiment tests all 2,000 record IDs once across five folds. All four methods have identical test-fold membership within each split protocol, as recorded in `supplement/split_fingerprints.json`. No new experiment was trained during repository preparation. Recomputed supplementary statistics are separate from immutable archived outputs.

## Three levels of reproduction

1. **Inspect and reaggregate archived results.** Run `audit_results.py`, `verify_predictions.py`, then `compare_paper.py`. These need only Python's standard library. `audit_summary.json` covers 60 fold partitions and 207 summaries; `prediction_verification.json` covers 360 independently recomputed text metrics.
2. **Recreate the splits with the experiment programs.** Install ML dependencies, run the launcher with `--prepare-only`, then use `compare_splits.py EXPERIMENT runs/EXPERIMENT`. It compares ordered IDs for every train/test fold and exits nonzero on differences. Saved splits remain the authority if a new environment gives different results. A prepared output is nonempty; use a fresh output directory for training.
3. **Retrain and infer.** Use a CUDA-capable environment, the intended Qwen base-model revision and the original CLI settings if available. The supplied environment is incomplete, so exact generated predictions are not guaranteed. Save `python -m pip freeze`, Python/CUDA/GPU versions, model revision, full command, seeds, timestamps and full logs alongside each new run. Do not describe source defaults as verified historical invocation settings.

## Example commands

From the root:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py company_contrastive --dry-run
python scripts/run_experiment.py company_contrastive --gpu 0 --prepare-only
python scripts/compare_splits.py company_contrastive runs/company_contrastive
python scripts/run_experiment.py company_contrastive --gpu 0 --output-dir runs/company_contrastive_full
python scripts/run_experiment.py random_ridge --gpu 0 --output-dir runs/random_ridge_full
python scripts/run_experiment.py industry_pairwise --gpu 0 --output-dir runs/industry_pairwise_full
```

The launcher accepts any of the 12 registry IDs. It resolves its own root from `__file__`, so invoking its absolute path from another working directory also works. User-provided `--data` and `--output-dir` paths are resolved relative to the calling shell's working directory. The program path, packaged default data, output default and company metadata are rooted at the copied repository. Direct `src/` usage remains supported with explicit `--data`, `--output-dir` and, when required, `--company-info`.

Model access uses the Transformers model identifier in each script (`Qwen/Qwen2.5-7B-Instruct` by default) or a local model directory. Model/adapters are excluded from this release. Fresh outputs can contain newly trained adapters; `runs/` and model weight extensions are ignored by Git.

## Split definitions

Random evaluation uses `StratifiedKFold` with shuffled paragraph indices, seed 42 and five folds by default. The scripts attempt joint strata and fall back to less granular label combinations when strata are too small. Consult archived `fold_distribution.csv` for the selected strategy. Company-disjoint evaluation uses the source's greedy company assignment balancing company counts, paragraph counts and label distributions; it is not a generic claim of GroupKFold. Industry-disjoint evaluation uses fixed industry groups embedded in the source; the five groups are not newly resampled for each run. The seed is an algorithm default, and the persisted fold memberships identify the actual archived split.

The overlap audit uses ticker for company identity, metadata column `產業` for industry, `pdf_url` for source-report identity, ID for record identity and exact `data` string equality for duplicate text. It does not measure semantic near-duplicates. Report-level separation is evaluated from URLs; it does not compare downloaded PDF binaries. The original source report PDFs are not included.

## What is not yet reproducible

- The original environment versions, model revision, full run commands, adapter weights and optimizer checkpoints are absent. Dependencies are inferred, unpinned and not validated by training here. GPU runtime was unavailable for retraining validation.
- The corrected manuscript dated 2026-10-22 resolves the previous Table 2 discrepancy. All 464 displayed numbers in Tables 2–4 now match at printed precision, including the 24 derived degradation/delta values. All 13 quantitative-description checks pass. Reference PDF/LaTeX hashes identify this version; no PDF is redistributed in this experiment archive. This numerical agreement does not establish exact retraining reproducibility.
- The competition system's Six-task, RoBERTa, voting, submissions and leaderboard evidence are not present. The full paper's competition results cannot be reconstructed from this directory.
- Company CSVs and embeddings were preserved, and fold uncertainty was recomputed from saved metric CSVs. Company metric evaluation has not all been independently reimplemented and verified against predictions.
- Five dependent folds do not establish population confidence, repeated-seed stability, or per-company rank-position uncertainty. Additional seeds, company/industry resampling and external validation would be new analyses.

No new license is assigned by this preparation. `data/`, saved fold files and prediction JSONs contain competition-derived text/labels; embeddings are also derived artifacts. The provided material does not establish permission to publicly redistribute all of those files. If publication requires a restricted-data version, remove protected content consistently across data, folds, predictions and caches, and retain ID-only memberships plus data-acquisition instructions.
