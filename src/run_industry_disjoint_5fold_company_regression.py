#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-7B-Instruct Frozen Paragraph Embedding
+ Company Mean Pooling
+ Ridge Regression

核心流程
--------
1. 每個 paragraph 獨立送入 frozen Qwen2.5-7B-Instruct。
2. 對該 paragraph 的 token hidden states 做 attention-mask mean pooling，
   得到一個 paragraph embedding。
3. 同一家公司的「全部 paragraph embeddings」取平均，
   得到唯一一個 company embedding。
4. 以 company embedding 分別訓練三個 Ridge regression：
   ECR / CER / BHCR。
5. 不使用任何 window，不存在 paragraphs-per-window。
6. RQ2 評估只計算：
   MAE / RMSE / Spearman rho / Kendall tau-b。

依賴
----
pip install torch transformers bitsandbytes accelerate pandas numpy scipy scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import gc
import json
import os
# Set CUDA visibility from --gpu before importing torch.
import sys
_gpu = next((v.split("=", 1)[1] for v in sys.argv[1:] if v.startswith("--gpu=")), "0")
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

"""
python src/run_industry_disjoint_5fold_company_regression.py \
  --data data/vpesg4k_train_val.json \
  --company-info data/company_ECR_CER_BHCR.csv \
  --output-dir runs/industry_ridge \
  --embedding-model Qwen/Qwen2.5-7B-Instruct \
  --gpu 1 \
  --max-length 1024 \
  --embedding-batch-size 1 \
  --ridge-alpha 10
"""

FIELDS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

LABEL_SPACE = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

INDICATORS = ["ECR", "CER", "BHCR"]
CER_DENOMINATOR = "evidence"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--company-info", type=Path, required=True,
                        help="包含 ticker 與產業欄位的公司層級 CSV。")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Frozen embedding backbone.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # 這裡的 max-length 是「單一 paragraph」的 token 上限。
    parser.add_argument("--max-length", type=int, default=1024)

    # 這只是 GPU inference batching；不會把多個 paragraphs 合併成一個 sample。
    parser.add_argument("--embedding-batch-size", type=int, default=1)

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge L2 regularization strength.",
    )

    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="忽略既有 paragraph_embeddings.npz，重新計算 embeddings。",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["data", "records", "items"]:
            if isinstance(obj.get(key), list):
                return obj[key]

    raise ValueError("輸入 JSON 必須是 list，或包含 data/records/items list。")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_id(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_label(value: Any, field: str) -> str:
    value = "N/A" if value is None else str(value).strip()
    if not value:
        value = "N/A"

    if field == "verification_timeline" and value == "longer_than_5_years":
        value = "more_than_5_years"

    if value not in LABEL_SPACE[field]:
        raise ValueError(f"{field} 出現非法標籤：{value}")

    return value


def normalize_gold(item: Dict[str, Any]) -> Dict[str, str]:
    gold = {
        field: normalize_label(item.get(field), field)
        for field in FIELDS
    }

    if gold["promise_status"] == "No":
        gold["verification_timeline"] = "N/A"
        gold["evidence_status"] = "N/A"
        gold["evidence_quality"] = "N/A"

    if gold["evidence_status"] in ["No", "N/A"]:
        gold["evidence_quality"] = "N/A"

    return gold


def validate_dataset(records: List[Dict[str, Any]]) -> None:
    required = ["id", "data", "ticker", *FIELDS]
    seen_ids = set()

    for index, item in enumerate(records):
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"第 {index} 筆缺少欄位：{missing}")

        item_id = normalize_id(item["id"])
        ticker = normalize_id(item["ticker"])

        if not item_id:
            raise ValueError(f"第 {index} 筆 id 為空。")
        if not ticker:
            raise ValueError(f"第 {index} 筆 ticker 為空。")
        if item_id in seen_ids:
            raise ValueError(f"重複 id：{item_id}")

        seen_ids.add(item_id)
        normalize_gold(item)


def group_by_company(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_company: Dict[str, List[Dict[str, Any]]] = {}

    for item in records:
        ticker = normalize_id(item["ticker"])
        by_company.setdefault(ticker, []).append(item)

    for ticker in by_company:
        by_company[ticker] = sorted(
            by_company[ticker],
            key=lambda x: normalize_id(x["id"]),
        )

    return by_company


def safe_ratio(numerator: int, denominator: int) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def compute_company_target(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    計算企業級 gold targets：

    ECR =
        N(Promise=Yes AND Evidence=Yes)
        / N(Promise=Yes)

    CER =
        N(Promise=Yes AND Evidence=Yes AND Quality=Clear)
        / N(Promise=Yes AND Evidence=Yes)

    SHCR =
        [N(already) + N(within_2_years)]
        / N(Promise=Yes)

    LHCR component =
        N(more_than_5_years)
        / N(Promise=Yes)

    BHCR =
        2 * SHCR * LHCR / (SHCR + LHCR)

    若 SHCR = LHCR = 0，則 BHCR = 0。

    between_2_and_5_years 視為中期承諾：
    不進 SHCR / LHCR 分子，但仍包含在 Promise=Yes 分母中。
    """
    promises = 0
    evidence_yes = 0
    clear = 0

    short_horizon = 0
    medium_horizon = 0
    long_horizon = 0

    for item in records:
        gold = normalize_gold(item)

        p_yes = gold["promise_status"] == "Yes"
        e_yes = p_yes and gold["evidence_status"] == "Yes"
        q_clear = e_yes and gold["evidence_quality"] == "Clear"

        timeline = gold["verification_timeline"]

        is_short = (
            p_yes
            and timeline in [
                "already",
                "within_2_years",
            ]
        )
        is_medium = (
            p_yes
            and timeline == "between_2_and_5_years"
        )
        is_long = (
            p_yes
            and timeline == "more_than_5_years"
        )

        promises += int(p_yes)
        evidence_yes += int(e_yes)
        clear += int(q_clear)

        short_horizon += int(is_short)
        medium_horizon += int(is_medium)
        long_horizon += int(is_long)

    cer_den = (
        evidence_yes
        if CER_DENOMINATOR == "evidence"
        else promises
    )

    ecr = safe_ratio(
        evidence_yes,
        promises,
    )
    cer = safe_ratio(
        clear,
        cer_den,
    )

    shcr = safe_ratio(
        short_horizon,
        promises,
    )
    lhcr_component = safe_ratio(
        long_horizon,
        promises,
    )

    if pd.isna(shcr) or pd.isna(lhcr_component):
        bhcr = np.nan
    elif shcr == 0.0 and lhcr_component == 0.0:
        bhcr = 0.0
    else:
        denominator = shcr + lhcr_component
        bhcr = (
            0.0
            if denominator == 0.0
            else (
                2.0
                * shcr
                * lhcr_component
                / denominator
            )
        )

    return {
        "n_paragraphs": len(records),
        "n_promises": promises,
        "n_evidence_yes": evidence_yes,
        "n_clear": clear,
        "n_short_horizon": short_horizon,
        "n_medium_horizon": medium_horizon,
        "n_long_horizon": long_horizon,
        "SHCR": shcr,
        "LHCR_component": lhcr_component,
        "ECR": ecr,
        "CER": cer,
        "BHCR": bhcr,
    }


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


@torch.no_grad()
def encode_all_paragraphs(
    records: List[Dict[str, Any]],
    model_name: str,
    max_length: int,
    batch_size: int,
    cache_path: Path,
    recompute: bool,
) -> Dict[str, np.ndarray]:
    """
    每個 paragraph 獨立 embedding。

    paragraph embedding:
      Qwen last_hidden_state
      -> attention-mask mean pooling across tokens
      -> L2 normalize

    完全沒有 paragraph windows。
    """
    expected_ids = [normalize_id(r["id"]) for r in records]

    if cache_path.exists() and not recompute:
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["ids"].astype(str).tolist()
        cached_embeddings = cached["embeddings"].astype(np.float32)

        if (
            len(cached_ids) == len(expected_ids)
            and set(cached_ids) == set(expected_ids)
        ):
            print(f"使用既有 paragraph embeddings：{cache_path}")
            return {
                pid: emb
                for pid, emb in zip(cached_ids, cached_embeddings)
            }

        print("Embedding cache 與目前 dataset 不一致，重新計算。")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )
    model.eval()

    if torch.cuda.is_available():
        device = next(model.parameters()).device
    else:
        device = torch.device("cpu")
        model.to(device)

    ids: List[str] = []
    embedding_batches: List[np.ndarray] = []

    for start in tqdm(
        range(0, len(records), batch_size),
        desc="Encoding paragraphs with frozen Qwen",
    ):
        batch_records = records[start:start + batch_size]
        texts = [
            str(item.get("data", "")).strip()
            for item in batch_records
        ]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        outputs = model(
            **encoded,
            return_dict=True,
        )

        hidden = outputs.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        # token-level mean pooling for each paragraph
        paragraph_embeddings = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1.0)
        )

        paragraph_embeddings = torch.nn.functional.normalize(
            paragraph_embeddings,
            p=2,
            dim=1,
        )

        embedding_batches.append(
            paragraph_embeddings.cpu().numpy().astype(np.float32)
        )
        ids.extend(
            normalize_id(item["id"])
            for item in batch_records
        )

    embeddings = np.concatenate(embedding_batches, axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=np.asarray(ids),
        embeddings=embeddings,
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        pid: emb
        for pid, emb in zip(ids, embeddings)
    }


def build_company_table(
    records: List[Dict[str, Any]],
    paragraph_embeddings: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    對每家公司：
      所有 paragraph embeddings -> mean pooling -> company embedding

    每家公司只有一個 feature vector。
    """
    rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []

    for ticker, company_records in sorted(
        group_by_company(records).items()
    ):
        target = compute_company_target(company_records)

        # joint regression 需要三個指標都有合法 target
        if any(pd.isna(target[ind]) for ind in INDICATORS):
            continue

        paragraph_vectors = np.stack(
            [
                paragraph_embeddings[normalize_id(item["id"])]
                for item in company_records
            ],
            axis=0,
        )

        # 這裡是真的把「該公司全部 paragraphs」一起平均
        company_vector = paragraph_vectors.mean(axis=0)
        company_vector = l2_normalize(company_vector)

        rows.append(
            {
                "ticker": ticker,
                "company": company_records[0].get("company", ""),
                **target,
            }
        )
        vectors.append(company_vector)

    if not rows:
        return (
            pd.DataFrame(),
            np.empty((0, 0), dtype=np.float32),
        )

    return pd.DataFrame(rows), np.stack(vectors, axis=0)


def fit_predict_ridge(
    train_df: pd.DataFrame,
    train_x: np.ndarray,
    test_df: pd.DataFrame,
    test_x: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """
    三個 independent Ridge regressors：
    ECR / CER / BHCR
    """
    result = test_df.copy()

    for indicator in INDICATORS:
        y_train = train_df[indicator].to_numpy(dtype=float)

        regressor = Pipeline(
            [
                ("standardize", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        regressor.fit(train_x, y_train)

        pred = regressor.predict(test_x)
        pred = np.clip(pred, 0.0, 1.0)

        result[f"gold_{indicator}"] = (
            test_df[indicator].to_numpy(dtype=float)
        )
        result[f"pred_{indicator}"] = pred
        result[f"{indicator}_error"] = (
            pred - result[f"gold_{indicator}"]
        )
        result[f"{indicator}_absolute_error"] = np.abs(
            result[f"{indicator}_error"]
        )

    return result


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    if (
        len(y_true) < 3
        or np.unique(y_true).size < 2
        or np.unique(y_pred).size < 2
    ):
        return np.nan, np.nan

    result = spearmanr(y_true, y_pred)
    return float(result.statistic), float(result.pvalue)


def evaluate_company_regression(
    company_df: pd.DataFrame,
    fold: Any,
) -> pd.DataFrame:
    """
    RQ2 只計算：
    - MAE
    - RMSE
    - Spearman rho
    - Kendall tau-b
    """
    rows = []

    for indicator in INDICATORS:
        valid = company_df[
            [f"gold_{indicator}", f"pred_{indicator}"]
        ].dropna()

        y_true = valid[f"gold_{indicator}"].to_numpy(dtype=float)
        y_pred = valid[f"pred_{indicator}"].to_numpy(dtype=float)

        if len(valid):
            error = y_pred - y_true
            mae = float(np.abs(error).mean())
            rmse = float(np.sqrt(np.square(error).mean()))
        else:
            mae = np.nan
            rmse = np.nan

        spearman_rho, spearman_p = safe_spearman(
            y_true,
            y_pred,
        )

        if (
            len(valid) >= 2
            and np.unique(y_true).size >= 2
            and np.unique(y_pred).size >= 2
        ):
            kt = kendalltau(
                y_true,
                y_pred,
                variant="b",
                nan_policy="omit",
            )
            kendall_tau_b = float(kt.statistic)
            kendall_p = float(kt.pvalue)
        else:
            kendall_tau_b = np.nan
            kendall_p = np.nan

        rows.append(
            {
                "fold": fold,
                "indicator": indicator,
                "n_companies": len(valid),
                "mae": mae,
                "rmse": rmse,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau_b": kendall_tau_b,
                "kendall_p": kendall_p,
                "n_spearman": len(valid),
            }
        )

    return pd.DataFrame(rows)


def build_mean_std(
    all_eval: pd.DataFrame,
) -> pd.DataFrame:
    return (
        all_eval
        .groupby("indicator", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            kendall_mean=("kendall_tau_b", "mean"),
            kendall_std=("kendall_tau_b", "std"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Industry-disjoint fixed 5-fold: "
            "Frozen Qwen paragraph embeddings + company mean pooling + Ridge"
        )
    )
    return add_common_args(parser).parse_args()



# =========================================================
# Fixed Industry-disjoint 5-fold
# =========================================================

INDUSTRY_FOLDS = {
    1: ["電腦及週邊設備業", "造紙工業"],
    2: ["電子零組件業", "航運業", "其他電子業", "鋼鐵工業", "光電業", "其他業"],
    3: ["金融保險業"],
    4: ["半導體業", "汽車工業", "食品工業"],
    5: ["通信網路業", "水泥工業", "塑膠工業", "油電燃氣業", "生技醫療業", "貿易百貨業"],
}

EXPECTED_FOLD_COUNTS = {
    1: {"companies": 8, "paragraphs": 345},
    2: {"companies": 10, "paragraphs": 370},
    3: {"companies": 12, "paragraphs": 551},
    4: {"companies": 9, "paragraphs": 377},
    5: {"companies": 10, "paragraphs": 357},
}

INDUSTRY_COLUMN_CANDIDATES = [
    "產業", "產業別", "industry", "Industry", "industry_name"
]


def load_company_industry_map(
    company_info_path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    df = pd.read_csv(
        company_info_path,
        dtype={"ticker": str},
        encoding="utf-8-sig",
    )

    if "ticker" not in df.columns:
        raise ValueError(f"{company_info_path} 缺少 ticker 欄位。")

    industry_col = next(
        (c for c in INDUSTRY_COLUMN_CANDIDATES if c in df.columns),
        None,
    )
    if industry_col is None:
        raise ValueError(
            f"公司資訊 CSV 找不到產業欄位，可接受：{INDUSTRY_COLUMN_CANDIDATES}"
        )

    df = df.copy()
    df["ticker"] = (
        df["ticker"].astype(str).str.strip().str.replace(r"\\.0$", "", regex=True)
    )
    df["industry"] = df[industry_col].astype(str).str.strip()

    if df["ticker"].duplicated().any():
        dup = sorted(df.loc[df["ticker"].duplicated(), "ticker"].unique().tolist())
        raise ValueError(f"公司資訊 CSV ticker 重複：{dup}")

    mapping = dict(zip(df["ticker"], df["industry"]))
    return mapping, df


def attach_industry(
    records: List[Dict[str, Any]],
    ticker_to_industry: Dict[str, str],
) -> List[Dict[str, Any]]:
    output = []
    for item in records:
        ticker = normalize_id(item["ticker"])
        if ticker not in ticker_to_industry:
            raise ValueError(f"ticker={ticker} 在 company-info CSV 找不到產業。")
        row = dict(item)
        row["industry"] = ticker_to_industry[ticker]
        output.append(row)
    return output


def validate_industry_definition(
    records: List[Dict[str, Any]],
) -> None:
    ticker_industries: Dict[str, set] = {}

    for item in records:
        ticker = normalize_id(item["ticker"])
        industry = str(item["industry"]).strip()
        ticker_industries.setdefault(ticker, set()).add(industry)

    inconsistent = {
        t: sorted(v)
        for t, v in ticker_industries.items()
        if len(v) != 1
    }
    if inconsistent:
        raise RuntimeError(f"同一 ticker 對應多個產業：{inconsistent}")

    configured = [
        industry
        for industries in INDUSTRY_FOLDS.values()
        for industry in industries
    ]
    if len(configured) != len(set(configured)):
        raise RuntimeError("INDUSTRY_FOLDS 中有重複產業。")

    dataset_industries = {str(item["industry"]).strip() for item in records}
    configured_industries = set(configured)

    missing = dataset_industries - configured_industries
    if missing:
        raise RuntimeError(f"以下產業未配置到任何 fold：{sorted(missing)}")


def build_industry_disjoint_folds(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    validate_industry_definition(records)

    folds = []
    summary_rows = []
    all_ids = {normalize_id(item["id"]) for item in records}
    seen_test_ids = set()

    for fold_number in range(1, 6):
        wanted_industries = set(INDUSTRY_FOLDS[fold_number])

        test_records = [
            item for item in records
            if str(item["industry"]).strip() in wanted_industries
        ]
        test_ids = {normalize_id(item["id"]) for item in test_records}
        train_records = [
            item for item in records
            if normalize_id(item["id"]) not in test_ids
        ]

        train_companies = {normalize_id(i["ticker"]) for i in train_records}
        test_companies = {normalize_id(i["ticker"]) for i in test_records}
        train_industries = {str(i["industry"]).strip() for i in train_records}
        test_industries = {str(i["industry"]).strip() for i in test_records}

        company_overlap = train_companies & test_companies
        industry_overlap = train_industries & test_industries

        if company_overlap:
            raise RuntimeError(
                f"Fold {fold_number} company leakage：{sorted(company_overlap)}"
            )
        if industry_overlap:
            raise RuntimeError(
                f"Fold {fold_number} industry leakage：{sorted(industry_overlap)}"
            )
        if test_industries != wanted_industries:
            raise RuntimeError(
                f"Fold {fold_number} 實際 test industries 與設定不一致。"
            )

        expected = EXPECTED_FOLD_COUNTS[fold_number]
        if len(test_companies) != expected["companies"]:
            raise RuntimeError(
                f"Fold {fold_number} 公司數={len(test_companies)}，"
                f"預期={expected['companies']}。"
            )
        if len(test_records) != expected["paragraphs"]:
            raise RuntimeError(
                f"Fold {fold_number} paragraphs={len(test_records)}，"
                f"預期={expected['paragraphs']}。"
            )

        if seen_test_ids & test_ids:
            raise RuntimeError(f"Fold {fold_number} 與前面 fold 有重複 test id。")
        seen_test_ids |= test_ids

        fold_index = fold_number - 1

        folds.append({
            "fold": fold_index,
            "fold_number": fold_number,
            "train_records": train_records,
            "test_records": test_records,
            "train_companies": sorted(train_companies),
            "test_companies": sorted(test_companies),
            "train_industries": sorted(train_industries),
            "test_industries": sorted(test_industries),
        })

        summary_rows.append({
            "fold": fold_index,
            "fold_number": fold_number,
            "n_train": len(train_records),
            "n_test": len(test_records),
            "n_train_companies": len(train_companies),
            "n_test_companies": len(test_companies),
            "n_train_industries": len(train_industries),
            "n_test_industries": len(test_industries),
            "test_industries": "|".join(INDUSTRY_FOLDS[fold_number]),
            "test_companies": "|".join(sorted(test_companies)),
        })

    if seen_test_ids != all_ids:
        raise RuntimeError("5 folds 未完整覆蓋全部資料。")

    return folds, pd.DataFrame(summary_rows)


def save_fold_files(
    folds: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    for fold in folds:
        fold_index = fold["fold"]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        save_json(fold["train_records"], fold_dir / "train.json")
        save_json(fold["test_records"], fold_dir / "test.json")
        save_json(
            {
                "fold": fold_index,
                "fold_number": fold["fold_number"],
                "split_type": "industry_disjoint_fixed",
                "train_samples": len(fold["train_records"]),
                "test_samples": len(fold["test_records"]),
                "train_companies": fold["train_companies"],
                "test_companies": fold["test_companies"],
                "train_industries": fold["train_industries"],
                "test_industries": fold["test_industries"],
                "company_overlap": [],
                "industry_overlap": [],
            },
            fold_dir / "split_info.json",
        )

def main() -> None:
    args = parse_args()

    if args.n_splits != 5:
        raise ValueError("固定 Industry-disjoint 實驗的 --n-splits 必須為 5。")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(args.data)
    validate_dataset(records)

    ticker_to_industry, _ = load_company_industry_map(
        args.company_info
    )
    records = attach_industry(
        records,
        ticker_to_industry,
    )

    folds, fold_summary = build_industry_disjoint_folds(
        records,
    )

    fold_summary.to_csv(
        args.output_dir / "fold_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_fold_files(
        folds,
        args.output_dir,
    )

    print("=" * 80)
    print(
        "Industry-disjoint fixed 5-fold | "
        "Qwen Embedding + Company Mean Pooling + Ridge"
    )
    print("=" * 80)
    print(f"資料筆數：{len(records)}")
    print(
        f"企業數：{len(group_by_company(records))}"
    )
    print(f"產業數：{len({item['industry'] for item in records})}")
    print(
        fold_summary[[
            "fold", "fold_number", "n_test_companies", "n_test",
            "n_test_industries", "test_industries"
        ]].to_string(index=False)
    )

    if args.prepare_only:
        return

    paragraph_embeddings = encode_all_paragraphs(
        records=records,
        model_name=args.embedding_model,
        max_length=args.max_length,
        batch_size=args.embedding_batch_size,
        cache_path=(
            args.output_dir
            / "paragraph_embeddings.npz"
        ),
        recompute=args.recompute_embeddings,
    )

    all_company_predictions = []
    all_evaluations = []

    for fold_index in range(len(folds)):
        fold_dir = args.output_dir / f"fold_{fold_index}"

        train_records = load_json(
            fold_dir / "train.json"
        )
        test_records = load_json(
            fold_dir / "test.json"
        )

        train_df, train_x = build_company_table(
            train_records,
            paragraph_embeddings,
        )
        test_df, test_x = build_company_table(
            test_records,
            paragraph_embeddings,
        )

        print("\n" + "=" * 80)
        print(f"Fold {fold_index}")
        print(
            f"Train companies={len(train_df)}, "
            f"Test companies={len(test_df)}"
        )
        print("=" * 80)

        if len(train_df) < 2:
            raise RuntimeError(
                f"Fold {fold_index} training companies 不足。"
            )
        if len(test_df) < 1:
            raise RuntimeError(
                f"Fold {fold_index} test companies 不足。"
            )

        company_df = fit_predict_ridge(
            train_df=train_df,
            train_x=train_x,
            test_df=test_df,
            test_x=test_x,
            alpha=args.ridge_alpha,
        )

        ticker_industry = {
            normalize_id(item["ticker"]): str(item["industry"]).strip()
            for item in test_records
        }
        company_df["industry"] = company_df["ticker"].map(ticker_industry)
        company_df["fold"] = fold_index
        company_df["fold_number"] = fold_index + 1

        eval_df = evaluate_company_regression(
            company_df,
            fold_index,
        )

        company_df.to_csv(
            fold_dir / "company_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        eval_df.to_csv(
            fold_dir / "regression_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_company_predictions.append(company_df)
        all_evaluations.append(eval_df)

    all_company = pd.concat(
        all_company_predictions,
        ignore_index=True,
    )
    all_eval = pd.concat(
        all_evaluations,
        ignore_index=True,
    )

    all_company.to_csv(
        args.output_dir / "fold_company_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_eval.to_csv(
        args.output_dir / "fold_regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_std = build_mean_std(all_eval)
    mean_std.to_csv(
        args.output_dir / "regression_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # industry-disjoint:
    # 每家公司與每個產業只出現在一個 test fold，因此可直接 OOF 合併。
    oof = all_company.copy()
    oof_eval = evaluate_company_regression(
        oof,
        "OOF_global",
    )

    # 額外輸出逐產業 OOF regression 指標
    industry_eval_rows = []
    for industry, group in oof.groupby("industry", sort=True):
        one = evaluate_company_regression(
            group,
            fold=f"industry::{industry}",
        )
        one.insert(0, "industry", industry)
        industry_eval_rows.append(one)

    if industry_eval_rows:
        pd.concat(
            industry_eval_rows,
            ignore_index=True,
        ).to_csv(
            args.output_dir / "oof_industry_regression_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )

    oof.to_csv(
        args.output_dir / "oof_company_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof_eval.to_csv(
        args.output_dir / "oof_regression_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n企業指標 fold mean ± std：")
    print(
        mean_std.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOOF 全企業結果：")
    print(
        oof_eval.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\n輸出資料夾："
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
