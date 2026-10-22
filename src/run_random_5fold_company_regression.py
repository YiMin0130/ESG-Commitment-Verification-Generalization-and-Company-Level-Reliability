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
python src/run_random_5fold_company_regression.py \
  --data data/vpesg4k_train_val.json \
  --output-dir runs/random_ridge \
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


from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random stratified 5-fold: "
            "Frozen Qwen paragraph embeddings + company mean pooling + Ridge"
        )
    )
    return add_common_args(parser).parse_args()


def make_stratification_labels(
    records: List[Dict[str, Any]],
    n_splits: int,
) -> Tuple[List[str], str]:
    gold_rows = [
        normalize_gold(item)
        for item in records
    ]

    candidates = [
        (
            "promise+timeline+evidence+quality",
            [
                "||".join(
                    [
                        g["promise_status"],
                        g["verification_timeline"],
                        g["evidence_status"],
                        g["evidence_quality"],
                    ]
                )
                for g in gold_rows
            ],
        ),
        (
            "promise+evidence+quality",
            [
                "||".join(
                    [
                        g["promise_status"],
                        g["evidence_status"],
                        g["evidence_quality"],
                    ]
                )
                for g in gold_rows
            ],
        ),
        (
            "promise+evidence",
            [
                "||".join(
                    [
                        g["promise_status"],
                        g["evidence_status"],
                    ]
                )
                for g in gold_rows
            ],
        ),
        (
            "promise",
            [
                g["promise_status"]
                for g in gold_rows
            ],
        ),
    ]

    for name, labels in candidates:
        if min(Counter(labels).values()) >= n_splits:
            return labels, name

    raise ValueError("無法建立 StratifiedKFold。")


def build_random_folds(
    records: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, str]:
    labels, strategy = make_stratification_labels(
        records,
        n_splits,
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    indices = np.arange(len(records))
    folds = []
    summary_rows = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(indices, labels)
    ):
        train_records = [
            records[int(i)]
            for i in train_indices
        ]
        test_records = [
            records[int(i)]
            for i in test_indices
        ]

        train_companies = {
            normalize_id(item["ticker"])
            for item in train_records
        }
        test_companies = {
            normalize_id(item["ticker"])
            for item in test_records
        }

        folds.append(
            {
                "fold": fold_index,
                "train_records": train_records,
                "test_records": test_records,
            }
        )

        summary_rows.append(
            {
                "fold": fold_index,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "n_train_companies": len(train_companies),
                "n_test_companies": len(test_companies),
                "n_overlapping_companies": len(
                    train_companies & test_companies
                ),
                "stratification_strategy": strategy,
            }
        )

    return folds, pd.DataFrame(summary_rows), strategy


def save_fold_files(
    folds: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    for fold in folds:
        fold_index = fold["fold"]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = fold["train_records"]
        test_records = fold["test_records"]

        train_companies = {
            normalize_id(item["ticker"])
            for item in train_records
        }
        test_companies = {
            normalize_id(item["ticker"])
            for item in test_records
        }

        save_json(
            train_records,
            fold_dir / "train.json",
        )
        save_json(
            test_records,
            fold_dir / "test.json",
        )
        save_json(
            {
                "fold": fold_index,
                "split_type": "random_stratified_seen_company",
                "train_samples": len(train_records),
                "test_samples": len(test_records),
                "train_companies": sorted(train_companies),
                "test_companies": sorted(test_companies),
                "overlapping_companies": sorted(
                    train_companies & test_companies
                ),
            },
            fold_dir / "split_info.json",
        )


def build_global_gold(
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for ticker, company_records in sorted(
        group_by_company(records).items()
    ):
        target = compute_company_target(
            company_records
        )

        if any(
            pd.isna(target[ind])
            for ind in INDICATORS
        ):
            continue

        rows.append(
            {
                "ticker": ticker,
                "company": (
                    company_records[0].get("company", "")
                ),
                **target,
            }
        )

    return pd.DataFrame(rows)


def aggregate_random_oof(
    all_company: pd.DataFrame,
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Random paragraph-level 5-fold 下，
    同一公司會在多個 test folds 出現。

    對各 fold 的 company prediction，
    依該 fold 中該公司的 test paragraph 數加權平均，
    再與完整資料的 company Gold 比較。

    注意：此 Random 設定是 seen-company diagnostic；
    RQ2 主結果建議仍以 company-disjoint 為主。
    """
    temp = all_company.copy()

    for indicator in INDICATORS:
        temp[f"weighted_pred_{indicator}"] = (
            temp[f"pred_{indicator}"]
            * temp["n_paragraphs"]
        )

    agg = (
        temp
        .groupby(
            ["ticker", "company"],
            as_index=False,
        )
        .agg(
            oof_test_paragraphs=(
                "n_paragraphs",
                "sum",
            ),
            weighted_pred_ECR=(
                "weighted_pred_ECR",
                "sum",
            ),
            weighted_pred_CER=(
                "weighted_pred_CER",
                "sum",
            ),
            weighted_pred_BHCR=(
                "weighted_pred_BHCR",
                "sum",
            ),
        )
    )

    for indicator in INDICATORS:
        agg[f"pred_{indicator}"] = (
            agg[f"weighted_pred_{indicator}"]
            / agg["oof_test_paragraphs"]
        )

    gold = build_global_gold(records).rename(
        columns={
            indicator: f"gold_{indicator}"
            for indicator in INDICATORS
        }
    )

    return gold.merge(
        agg[
            [
                "ticker",
                "company",
                "oof_test_paragraphs",
                "pred_ECR",
                "pred_CER",
                "pred_BHCR",
            ]
        ],
        on=["ticker", "company"],
        how="inner",
        validate="one_to_one",
    )


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(args.data)
    validate_dataset(records)

    folds, fold_summary, strategy = build_random_folds(
        records,
        args.n_splits,
        args.seed,
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
        "Random stratified 5-fold | "
        "Qwen Embedding + Company Mean Pooling + Ridge"
    )
    print("=" * 80)
    print(f"資料筆數：{len(records)}")
    print(
        f"企業數：{len(group_by_company(records))}"
    )
    print(
        f"Stratification strategy: {strategy}"
    )
    print(fold_summary.to_string(index=False))

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

    for fold_index in range(args.n_splits):
        fold_dir = args.output_dir / f"fold_{fold_index}"

        train_records = load_json(
            fold_dir / "train.json"
        )
        test_records = load_json(
            fold_dir / "test.json"
        )

        # Random setting 下：
        # train company embedding 只平均 train paragraphs；
        # test company embedding 只平均 test paragraphs。
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
        company_df["fold"] = fold_index

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

    oof = aggregate_random_oof(
        all_company,
        records,
    )
    oof_eval = evaluate_company_regression(
        oof,
        "OOF_global",
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
