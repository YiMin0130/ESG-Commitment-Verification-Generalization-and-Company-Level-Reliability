# Data and artifact formats

The ordered input contains 2,000 Traditional Chinese paragraph records. The supplied paper identifies them as the labeled training/validation subset of the 4,000-instance VeriPromiseESG4K corpus, covering 2024 reports. This archive has 49 tickers and 18 industries. The audit derives counts directly from the records and metadata.

## Input fields

| Field | Meaning |
|---|---|
| `id` | Paragraph record identifier, treated as a string |
| `data` | Paragraph text |
| `ticker`, `company` | Company identifier and source company name |
| `pdf_url`, `page_number`, `company_source` | Source report and provenance |
| `esg_type` | Supplied E/S/G category |
| `promise_status` | `Yes`, `No` |
| `verification_timeline` | `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years`, `N/A` |
| `evidence_status` | `Yes`, `No`, `N/A` |
| `evidence_quality` | `Clear`, `Not Clear`, `Misleading`, `N/A` |
| `promise_string`, `evidence_string` | Original annotated spans, retained in input |
| `industry` | Added in some experiment exports from metadata; not required in the reconstructed base file |

When no promise exists, the other three tasks are not applicable. When evidence is absent/not applicable, evidence quality is not applicable. The programs normalize labels and apply conditional rules; the source code preserves the original implementation. Classification macro-F1 uses the complete configured label set with zero division set to zero. Balanced accuracy averages recall over classes present in the gold labels. Rare-class support is available in archived `per_class_metrics.csv` and in supplementary label distributions.

`company_ECR_CER_BHCR.csv` contains ticker, company, counts, derived indicators, percentage columns and `產業` (industry). Programs use its ticker/industry mapping for industry splits. The CSV's rounded indicator values should not replace full-precision ratios computed from paragraph labels in an experiment.

## Indicator definitions

For each company and evaluated paragraph subset, let P be the count with promise=Yes, E the count with promise=Yes and evidence=Yes, and C the count with promise=Yes, evidence=Yes and quality=Clear.

- ECR = E / P.
- CER = C / E. The code uses the evidence denominator, not all promises.
- SHCR = count(promise=Yes and timeline in {already, within_2_years}) / P.
- LHCR component = count(promise=Yes and timeline=more_than_5_years) / P.
- BHCR = 2 × SHCR × LHCR / (SHCR + LHCR), with zero when both horizon components are zero.

Medium-horizon commitments remain in P but enter neither horizon numerator. A zero ratio denominator is handled as NaN by `safe_ratio`; inspect evaluation CSV sample counts because undefined values can be excluded. These describe evidence coverage, clarity and horizon balance in the labels; they are not independently measured company outcomes.

Random-fold company outputs aggregate the paragraphs in that fold's test subset, rather than each company's complete corpus. Thus the same company can have different gold indicators in different random folds. In group-disjoint evaluation, all held-out paragraphs for a company are held out together. OOF-global and fold-level aggregations must not be combined as interchangeable evaluations.

## Results and supplementary schemas

| Artifact | Contents |
|---|---|
| `fold_i/train.json`, `test.json` | Exact saved records for each partition |
| `fold_i/split_info.json` | Fold metadata; some random programs also save original positional indices |
| `predictions.json` | ID, text, gold/predicted fields and raw generation where supplied |
| `classification_summary.csv`, `per_class_metrics.csv` | Task metrics and label-level support/precision/recall/F1 |
| `company_level_metrics.csv`, `company_predictions.csv` | Gold/predicted company indicators and errors, depending on method |
| `company_indicator_evaluation.csv`, `regression_evaluation.csv` | Company MAE/RMSE and rank agreement |
| Pairwise CSVs | Training pairs, test pair preferences, accumulated company rankings and correlations; names vary by experiment |
| `fold_*_results.csv` | Fold observations used for supplementary mean/SD/interval calculation |
| `oof_*.csv`, `oof_predictions.json` | Original out-of-fold combined outputs where supplied |
| `paragraph_embeddings.npz` | Compressed paragraph embedding cache for frozen methods; retained, not model weights |
| `supplement/fold_metrics.csv` | Long-form experiment/split/method/fold/scope/target/metric/value/source/n |
| `mean_sd_descriptive_intervals.csv` | Five-fold mean, sample SD, variance and untruncated descriptive t interval |
| `split_membership.csv` | Experiment/fold/train-or-test/record ID/ticker/industry/hashed report URL; source folds retain full URLs |
| `split_overlap.csv` | Counts of unique entities and train/test intersections per fold |
| `file_manifest.csv` | Repository-relative file paths, byte lengths and SHA-256 hashes; excludes itself and Python caches |

`split_fingerprints.json` hashes sorted test IDs for each fold and is useful for confirming membership, not training order. Use `compare_splits.py` for ordered train/test comparison. Existing txt logs are historical console output, rather than comprehensive run manifests.
