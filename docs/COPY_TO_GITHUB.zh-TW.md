# 複製公開整理版

ZIP 內容即為專案根目錄。先解壓縮，切換到解壓後的目錄，再執行：

```powershell
python scripts/check_public_release.py
python scripts/audit_results.py
python scripts/verify_predictions.py
python scripts/compare_paper.py
```

若要複製到同層的新 repository 目錄，在目前目錄執行：

```powershell
$releaseSource = Get-Location
$repositoryTarget = Join-Path (Split-Path -Parent $releaseSource.Path) 'ESG-Commitment-Verification'
if (Test-Path -LiteralPath $repositoryTarget) { throw '目的地已存在，請使用新的空目錄。' }
Copy-Item -LiteralPath $releaseSource.Path -Destination $repositoryTarget -Recurse -Force
Set-Location -LiteralPath $repositoryTarget
Get-ChildItem -Force
```

範例沒有固定磁碟代號、個人帳號或工作站位置；執行時依你目前目錄計算目的地。`-Force` 包含 `.gitignore`。保留 src、configs、data、results、scripts、supplement、docs 及兩份 README 的相對位置。

```powershell
python scripts/run_experiment.py industry_qlora --dry-run
python scripts/run_experiment.py industry_qlora --gpu 0 --output-dir runs/industry_qlora_full --base-model models/Qwen2.5-7B-Instruct
```

預設也可使用模型的公開名稱；本機模型相對路徑以專案根目錄為準。Embedding 方法改用 `--embedding-model models/Qwen2.5-7B-Instruct`。新輸出放在 runs，避免覆寫 results。

確認資料公開權限後，可自行執行 git init、git add、git commit，並依 GitHub 提供的命令設定 remote 與 push。本整理沒有上傳檔案。Tables 2–4 共 464 個數值與 13 項定量描述已核對一致；尚未進行 GPU 重訓。論文及隊伍名稱仍可用來辨認公開研究，企業文本／來源網址／公開信箱則保留為研究資料。隱私掃描不代表競賽資料再散布授權已取得。
