# ESG 承諾驗證

[English](README.md) | **繁體中文**

本專案提供 TEAM_9924 論文 **Evaluating Generalization and Company-Level Reliability in ESG Commitment Verification** 的程式碼與實驗結果。

研究以 Qwen2.5-7B-Instruct 評估 ESG 承諾驗證在已見公司、未見公司與未見產業間的泛化能力，並透過 ECR、CER 與 BHCR 比較公司層級的指標估計與排序表現。

專案包含 12 組實驗、每組五折的資料切分、段落預測、公司層級輸出與詳細結果。標註資料涵蓋 49 家公司、18 個產業，共 2,000 筆繁體中文段落。模型權重與訓練完成的 adapter 需另外取得或訓練。

## 實驗方法

| 方法 | 已見公司 | 未見公司 | 未見產業 |
|---|---|---|---|
| QLoRA | `random_qlora` | `company_qlora` | `industry_qlora` |
| 對比學習 QLoRA | `random_contrastive` | `company_contrastive` | `industry_contrastive` |
| Frozen Qwen embedding + Ridge | `random_ridge` | `company_ridge` | `industry_ridge` |
| Frozen Qwen embedding + 成對排序 | `random_pairwise` | `company_pairwise` | `industry_pairwise` |

四個段落分類任務為 `promise_status`、`verification_timeline`、`evidence_status` 與 `evidence_quality`。公司指標分別描述證據覆蓋率（ECR）、證據清晰度（CER）及承諾時間跨度的平衡程度（BHCR）。

## 專案結構

```text
configs/       實驗定義與預設參數
data/          標註段落與公司／產業資訊
src/           實驗程式
results/       依切分方式及方法分類的既有結果
supplement/    詳細指標、分布與逐折資料成員
scripts/       實驗執行入口與分析工具
docs/          資料格式與重現說明
runs/          執行新實驗時建立的輸出目錄
```

`results/` 中每組實驗包含 `fold_0` 至 `fold_4`，保留訓練／測試資料、切分資訊與各方法的輸出。

## 環境安裝

使用 Python 3.10 以上版本，模型訓練與推論需要支援 CUDA 的 GPU。先安裝符合 CUDA 環境的 PyTorch，再安裝專案套件。

```bash
python -m venv .venv
```

啟用虛擬環境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

在專案根目錄安裝套件：

```bash
python -m pip install -r requirements.txt
```

## 執行實驗

列出可執行的實驗：

```bash
python scripts/run_experiment.py --list
```

使用預設基底模型 `Qwen/Qwen2.5-7B-Instruct` 執行：

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --output-dir runs/industry_qlora
python scripts/run_experiment.py random_ridge --gpu 0 --output-dir runs/random_ridge
python scripts/run_experiment.py company_pairwise --gpu 0 --output-dir runs/company_pairwise
```

僅建立訓練／測試切分：

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --prepare-only --output-dir runs/industry_splits
```

指定本機模型目錄：

```bash
python scripts/run_experiment.py industry_qlora --gpu 0 --output-dir runs/industry_qlora_local --base-model models/Qwen2.5-7B-Instruct
python scripts/run_experiment.py random_ridge --gpu 0 --output-dir runs/random_ridge_local --embedding-model models/Qwen2.5-7B-Instruct
```

以上命令在專案根目錄執行。入口程式依專案位置尋找資料與程式，並在需要時帶入公司／產業資訊。每次執行請指定新的空輸出目錄。其他參數可直接接在命令後方，完整列表見[實驗定義](configs/experiments.json)。

## 預設參數

| 設定 | 預設值 |
|---|---|
| 交叉驗證 | 五折，seed 42 |
| QLoRA 訓練 | 3 epochs，learning rate 1e-4 |
| QLoRA 輸入長度 | 4096 tokens |
| Batch size／梯度累積 | 1／4 |
| LoRA | r=16、alpha=32、dropout=0.05 |
| 對比學習 | lambda=0.10、temperature=0.07 |
| Embedding 輸入長度 | 1024 tokens |
| Ridge alpha | 10 |

## 結果與資料

- [完整結果表](supplement/RESULTS.md)
- [各折評估指標](supplement/fold_metrics.csv)
- [平均、標準差與描述性區間](supplement/mean_sd_descriptive_intervals.csv)
- [逐折資料成員](supplement/split_membership.csv)
- [公司分布](supplement/company_distribution.csv)與[產業分布](supplement/industry_distribution.csv)
- [資料格式與指標公式](docs/DATA_AND_FORMATS.md)
- [重現說明](docs/REPRODUCIBILITY.md)
