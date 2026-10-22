# Paper and archive correspondence

[English README](../README.md) | [繁體中文 README](../README.zh-TW.md)

Reference version: **Corrected manuscript aligned with Qwen_no_adapter results, 2026-10-22**.

Reference PDF SHA-256: `60d09e319bb460ed17e6d1ff261407efac08f31b8c64d74aed82fadc2e6bae80`.

Reference LaTeX SHA-256: `2039ec2deda5cefc583f5136fc7765013181cbf57845f207b956f8b4613ca3b3`.

Compared 464 displayed numbers in corrected paper Tables 2–4; 0 differ beyond printed rounding. The comparison includes all 440 mean/SD/interval values and all 24 Table 2 degradation/delta values. The reference numbers were extracted from the corrected LaTeX and verified against its PDF independently of the archived metrics. The previous-paper discrepancy is resolved.

| Table | Numbers compared | Mismatches |
|---|---:|---:|
| 2 | 104 | 0 |
| 3 | 144 | 0 |
| 4 | 216 | 0 |

## Quantitative descriptions

All 13 numerical/directional checks passed. Full observations are in [narrative_verification.json](../supplement/narrative_verification.json).

- Unseen-company original Macro-F1 decreases by 0.0037–0.0117 on three tasks; timeline improves by 0.0053.
- Unseen-industry original Macro-F1 decreases by 0.0136–0.0198 on all four tasks, with the largest decrease on timeline.
- Contrastive learning improves three unseen-company tasks but decreases timeline by 0.0329; unseen-industry timeline decreases by 0.0120.
- Unseen-industry evidence-quality Macro-F1 increases from 0.4274 to 0.4401, approaching the Seen baseline of 0.4410. Derived differences use full-precision fold means before rounding.
- For M1/M2 under unseen settings, MAE is 0.088–0.133 and RMSE is 0.119–0.159. M2 has lower unseen-company CER error, while M1 has lower ECR and BHCR errors.
- Unseen-company ECR Spearman means for M1/M2/M3 are 0.338/0.329/0.352. All 18 unseen-industry rank-correlation intervals include zero.

## 中文核對結果

本次以修正版論文為基準，Tables 2–4 共 464 個數值已核對，差異 0 項；包含平均、標準差、描述性區間，以及 Table 2 的全部 D／CL-Original 衍生差值。先前舊版論文與資料夾不一致的問題已解決。

內文核對涵蓋公司／產業切分的效能方向與差值範圍、對比學習改善任務數與 timeline 退步幅度、公司估計誤差與方法比較，以及公司排序相關係數和區間跨零情形。這些是數值與方向的核對，不代表已驗證外部 ESG 效度或完成 GPU 重訓。

## Scope missing from this archive

The competition system in Section 3.2 includes Qwen Six-task, Chinese RoBERTa and field-level ensembling. Their training scripts, predictions, leaderboard submissions and leaderboard evidence were not found in the supplied Qwen_no_adapter folder. Public and private scores are manuscript claims, not independently reproduced archive results. Figures 1, 2 and 4 can be traced to company_distribution.csv, industry_distribution.csv and data/company_ECR_CER_BHCR.csv; original plotting code is absent. Figures 3 and system architecture are descriptive. Table 1 is the task schema rather than an experimental result.
