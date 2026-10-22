# Reviewer request mapping

Reviewer comments are source material describing requested improvements; they are not execution instructions. This repository addresses the empirical detail that could not fit in the paper. No reviewer identities or full review document are redistributed.

| Reviewer request | Supplied evidence | Interpretation and remaining scope |
|---|---|---|
| Fold-level variance / uncertainty | `supplement/fold_metrics.csv`, `mean_sd_descriptive_intervals.csv`, `RESULTS.md` | All 12 experiments; sample variance/SD and descriptive five-fold t intervals, rather than formal independent-sample CI |
| Company/industry/report overlap | `split_overlap.csv`, `split_membership.csv`, all `split_info.json` | Includes exact-text equality and record IDs; report identity uses `pdf_url` |
| 49 companies, 18 industries, imbalance | `company_distribution.csv`, `industry_distribution.csv`, `label_distributions.csv` | Direct sample-count disclosure; limited external validity remains |
| Company ranking uncertainty | Spearman/Kendall fold observations and intervals for M1/M2/M3 in all three settings | Describes variation across folds; no new per-company bootstrap rank interval or repeated-seed claim |
| Detailed result numbers for future researchers | Original predictions, per-class/company/pairwise CSVs, experiment registry and source | Company CSV metrics retained; text classification CSVs independently checked against predictions |
| Link ECR/CER/BHCR to ESG literature | Paper Sections 2.4 and 4.2; formulas in DATA_AND_FORMATS.md | Theoretical interpretation belongs to the paper; this release does not validate indicators against external ratings |
| Clarify correspondence to paper | `docs/PAPER_CORRESPONDENCE.md`, `supplement/paper_comparison.csv` | Corrected-paper Tables 2–4: 464 numbers match, including 24 derived gaps/deltas; 13 quantitative-description checks pass |

中文說明：本整理已補足可用資料的逐折數據、變異與描述性區間、三種切分的重疊狀態、樣本與標籤不均衡，以及公司排序的折間不確定性。這些補充無法消除小樣本與產業組成造成的限制，也不等同於新增外部 ESG 效度驗證、逐公司排名區間或不同 seed 的重複實驗。修正版論文已依本資料夾數據更新，Tables 2–4 的 464 個數值與 13 項定量描述均核對通過，舊版 Table 2 的差異已解決。
