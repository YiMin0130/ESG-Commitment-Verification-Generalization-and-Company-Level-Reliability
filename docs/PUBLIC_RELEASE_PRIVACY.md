# Public release privacy scope

[English README](../README.md) | [繁體中文 README](../README.zh-TW.md)

## Changes made

- Removed the original machine, user and project-directory prefixes from all 12 historical console logs. Output paths now refer to the corresponding repository-relative `results/` locations. Metric values and other log content were preserved.
- Replaced computer-specific copy instructions with commands based on the current directory. Model examples use a repository-relative `models/Qwen2.5-7B-Instruct` directory.
- Changed launcher command display to use `python` and repository-relative paths. External absolute arguments are displayed as a neutral placeholder; actual invocation still receives the user-selected path.
- Retained all split memberships, paragraph text, labels, predictions, metric CSVs and embedding caches. Six NPZ caches were loaded with pickling disabled and inspected: their only arrays contain numeric record-ID strings and floating-point embeddings. No machine/user path fields were present.
- Rebuilt the checksum manifest after cleanup. The public ZIP uses a fixed timestamp, neutral file permissions, no archive/file comments and no extra metadata fields. It includes no Git history, account configuration, model weights or reviewer documents.

## Path behavior

The programs contain no fixed private computer paths. The launcher derives the project root from its own script location and constructs data/config/source/output locations from that root. Absolute paths calculated in memory at runtime are normal for locating a copied project; they are not a dependency on the original machine. Relative user arguments resolve as described in REPRODUCIBILITY.md, and base-model examples are relative to the project root. Moving the whole directory preserves its internal links.

## What remains intentionally public

The paper title, team attribution and original company identifiers remain. Linking this repository to the paper can identify its authors. Corporate report excerpts may mention corporate laboratories, public contact addresses and report URLs; these are source data, not private research-workstation details. Removing them would alter inputs, reported results and report-overlap checks. Privacy preparation does not grant dataset redistribution rights.

The scan checks private machine path formats, common credential patterns, private IP ranges, unexpected binary files and cache headers. It cannot prove that arbitrary text contains no indirectly identifying fact, nor provide a guarantee against every possible disclosure. It does not remove the research's public authorship. New logs, new input data or new files should be checked again before release.

## Verify a modified copy

```bash
python scripts/check_public_release.py
python scripts/audit_results.py
python scripts/verify_predictions.py
python scripts/compare_paper.py
```

The final two commands continue to verify 360 saved text metrics and all 464 paper-table numbers plus 13 quantitative descriptions. Run the audit again last when refreshing file checksums after generated reports change. GPU retraining is outside this check.

## 中文說明

本次清除了 12 份歷史 log 的原電腦／使用者路徑與文件中的個人磁碟位置；複製範例、模型範例和執行入口顯示改為相對路徑。程式在執行時依所在位置動態定位專案，因此整個目錄可搬移使用。原始資料、標籤、預測、切分、數值 CSV 與六份 embedding 均保留。

公開企業文本中的信箱、來源網址及企業實驗室文字仍保留，論文名稱與隊伍名稱也仍可連結到作者；這份整理不會讓研究作者匿名。掃描沒有發現剩餘的私有路徑、常見憑證格式或私有 IP，但無法保證任意內容在所有情境下都不會被間接辨認。競賽資料公開權限仍依原有條款處理。

分享時請使用本次清理後的公開版 ZIP。較早產生的備份 ZIP 不會因資料夾更新而自動更新。
