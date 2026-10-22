#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Company-disjoint 5-fold
Frozen Qwen2.5-7B-Instruct Paragraph Embedding
+ Company Mean Pooling
+ PRP-inspired Pairwise Company Ranking

核心流程
--------
1. 每個 paragraph 獨立送入 frozen Qwen2.5-7B-Instruct。
2. 對 token hidden states 做 attention-mask mean pooling，得到 paragraph embedding。
3. 同一家公司的全部 paragraph embeddings 平均，得到 company mean embedding。
4. 先以 company-disjoint 方式切 fold，確保 train/test 無企業重疊。
5. 對每個 indicator (ECR / CER / BHCR)：
   - training companies 兩兩配對
   - pair feature = embedding_A - embedding_B
   - label = 1 if gold_A > gold_B else 0
   - gold 相同的 pair 不加入 training
   - 每個 unordered pair 同時加入 A-B 與 B-A
   - StandardScaler + LogisticRegression 訓練 pairwise ranker
6. test companies 兩兩比較，得到 winner。
7. 使用 All-pair win aggregation 得到 company ranking score。
8. 評估只計算：
   - Pairwise Accuracy
   - Spearman rho
   - Kendall tau-b

重要：
- split 是 company-disjoint，train/test 公司完全不重疊。
- Pairwise ranking score 是 fold 內相對排序分數，不是指標數值預測。
- 不計算 MAE / RMSE。
- 不建立跨 fold global OOF ranking，因為不同 fold 之間沒有 pairwise comparison，
  fold 內的 win-rate 不能直接當成全 49 家公司的共同全域排序分數。
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

"""
python src/run_company_disjoint_5fold_company_pairwise_ranking.py \
  --data data/vpesg4k_train_val.json \
  --output-dir runs/company_pairwise \
  --embedding-model Qwen/Qwen2.5-7B-Instruct \
  --gpu 1 \
  --max-length 1024 \
  --embedding-batch-size 1 \
  --logistic-c 1.0 \
  --max-iter 5000
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
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=1)

    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for LogisticRegression.",
    )

    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Ignore cached paragraph_embeddings.npz and recompute.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Company-disjoint 5-fold: "
            "Frozen Qwen paragraph embeddings + company mean pooling "
            "+ PRP-inspired pairwise ranking"
        )
    )
    return add_common_args(parser).parse_args()


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
    計算企業級 gold targets。

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

    若 SHCR = 0 且 LHCR = 0，則 BHCR = 0。

    between_2_and_5_years 視為中期承諾：
    不列入 SHCR 或 LHCR 分子，但若 Promise=Yes，
    仍包含在 Promise 分母中。
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
        texts = [str(item.get("data", "")).strip() for item in batch_records]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        outputs = model(**encoded, return_dict=True)

        hidden = outputs.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()

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
        ids.extend(normalize_id(item["id"]) for item in batch_records)

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

    return {pid: emb for pid, emb in zip(ids, embeddings)}


def build_company_table(
    records: List[Dict[str, Any]],
    paragraph_embeddings: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    每家公司：
      全部 paragraphs embeddings -> mean pooling -> L2 normalize

    只有 ECR/CER/BHCR 三個 target 都有效的公司會保留。
    """
    rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []

    for ticker, company_records in sorted(group_by_company(records).items()):
        target = compute_company_target(company_records)

        if any(pd.isna(target[ind]) for ind in INDICATORS):
            continue

        paragraph_vectors = np.stack(
            [
                paragraph_embeddings[normalize_id(item["id"])]
                for item in company_records
            ],
            axis=0,
        )

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
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32)

    return pd.DataFrame(rows), np.stack(vectors, axis=0)


def build_pairwise_training_data(
    company_df: pd.DataFrame,
    company_x: np.ndarray,
    indicator: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    unordered pair (i, j):
      若 gold_i == gold_j -> 排除
      否則同時加入：
        x_i - x_j, label(i > j)
        x_j - x_i, label(j > i)
    """
    x_pairs: List[np.ndarray] = []
    y_pairs: List[int] = []
    rows: List[Dict[str, Any]] = []

    gold = company_df[indicator].to_numpy(dtype=float)

    for i in range(len(company_df)):
        for j in range(i + 1, len(company_df)):
            yi = gold[i]
            yj = gold[j]

            if np.isclose(yi, yj):
                continue

            label_ij = int(yi > yj)
            diff_ij = company_x[i] - company_x[j]

            x_pairs.append(diff_ij)
            y_pairs.append(label_ij)
            rows.append(
                {
                    "indicator": indicator,
                    "ticker_A": company_df.iloc[i]["ticker"],
                    "ticker_B": company_df.iloc[j]["ticker"],
                    "gold_A": yi,
                    "gold_B": yj,
                    "label_A_gt_B": label_ij,
                }
            )

            x_pairs.append(-diff_ij)
            y_pairs.append(1 - label_ij)
            rows.append(
                {
                    "indicator": indicator,
                    "ticker_A": company_df.iloc[j]["ticker"],
                    "ticker_B": company_df.iloc[i]["ticker"],
                    "gold_A": yj,
                    "gold_B": yi,
                    "label_A_gt_B": 1 - label_ij,
                }
            )

    if not x_pairs:
        raise RuntimeError(
            f"{indicator}: training fold 無可用 pair，可能所有公司 gold 都相同。"
        )

    x_arr = np.stack(x_pairs, axis=0).astype(np.float32)
    y_arr = np.asarray(y_pairs, dtype=np.int64)

    if np.unique(y_arr).size < 2:
        raise RuntimeError(
            f"{indicator}: pairwise training labels 只有單一類別。"
        )

    return x_arr, y_arr, pd.DataFrame(rows)


def fit_pairwise_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    logistic_c: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=logistic_c,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def rank_test_companies_allpair(
    company_df: pd.DataFrame,
    company_x: np.ndarray,
    indicator: str,
    model: Pipeline,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    test fold 內全部 unordered company pairs 比較。
    winner +1 win，最後 wins/(N-1) 作為 fold 內 ranking score。
    """
    n = len(company_df)

    if n < 2:
        raise RuntimeError(
            f"{indicator}: test companies 少於 2，無法進行 pairwise ranking。"
        )

    wins = np.zeros(n, dtype=float)
    pair_rows: List[Dict[str, Any]] = []

    gold = company_df[indicator].to_numpy(dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            diff = (company_x[i] - company_x[j]).reshape(1, -1)
            prob_i_gt_j = float(model.predict_proba(diff)[0, 1])
            pred_i_gt_j = int(prob_i_gt_j >= 0.5)

            if pred_i_gt_j == 1:
                wins[i] += 1.0
                pred_winner = company_df.iloc[i]["ticker"]
            else:
                wins[j] += 1.0
                pred_winner = company_df.iloc[j]["ticker"]

            gold_tie = bool(np.isclose(gold[i], gold[j]))
            gold_i_gt_j = None if gold_tie else int(gold[i] > gold[j])
            pair_correct = (
                None
                if gold_tie
                else int(pred_i_gt_j == gold_i_gt_j)
            )

            pair_rows.append(
                {
                    "indicator": indicator,
                    "ticker_A": company_df.iloc[i]["ticker"],
                    "ticker_B": company_df.iloc[j]["ticker"],
                    "gold_A": gold[i],
                    "gold_B": gold[j],
                    "gold_tie": gold_tie,
                    "gold_A_gt_B": gold_i_gt_j,
                    "prob_A_gt_B": prob_i_gt_j,
                    "pred_A_gt_B": pred_i_gt_j,
                    "pred_winner": pred_winner,
                    "pair_correct": pair_correct,
                }
            )

    result = company_df.copy()
    result[f"gold_{indicator}"] = gold
    result[f"wins_{indicator}"] = wins
    result[f"rank_score_{indicator}"] = wins / float(n - 1)

    result[f"pred_rank_{indicator}"] = (
        pd.Series(result[f"rank_score_{indicator}"])
        .rank(method="average", ascending=False)
        .to_numpy()
    )
    result[f"gold_rank_{indicator}"] = (
        pd.Series(result[f"gold_{indicator}"])
        .rank(method="average", ascending=False)
        .to_numpy()
    )

    return result, pd.DataFrame(pair_rows)


def safe_spearman(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    if (
        len(y_true) < 3
        or np.unique(y_true).size < 2
        or np.unique(y_score).size < 2
    ):
        return np.nan, np.nan

    result = spearmanr(y_true, y_score)
    return float(result.statistic), float(result.pvalue)


def evaluate_pairwise_ranking(
    ranked_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    indicator: str,
    fold: Any,
) -> pd.DataFrame:
    valid_pairs = pair_df.loc[~pair_df["gold_tie"]].copy()

    pairwise_accuracy = (
        float(valid_pairs["pair_correct"].astype(float).mean())
        if len(valid_pairs)
        else np.nan
    )

    y_true = ranked_df[f"gold_{indicator}"].to_numpy(dtype=float)
    y_score = ranked_df[f"rank_score_{indicator}"].to_numpy(dtype=float)

    spearman_rho, spearman_p = safe_spearman(y_true, y_score)

    if (
        len(y_true) >= 2
        and np.unique(y_true).size >= 2
        and np.unique(y_score).size >= 2
    ):
        kt = kendalltau(
            y_true,
            y_score,
            variant="b",
            nan_policy="omit",
        )
        kendall_tau_b = float(kt.statistic)
        kendall_p = float(kt.pvalue)
    else:
        kendall_tau_b = np.nan
        kendall_p = np.nan

    return pd.DataFrame(
        [
            {
                "fold": fold,
                "indicator": indicator,
                "n_companies": len(ranked_df),
                "n_all_pairs": len(pair_df),
                "n_evaluable_pairs": len(valid_pairs),
                "n_gold_tie_pairs": int(pair_df["gold_tie"].sum()),
                "pairwise_accuracy": pairwise_accuracy,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "kendall_tau_b": kendall_tau_b,
                "kendall_p": kendall_p,
            }
        ]
    )


def build_mean_std(all_eval: pd.DataFrame) -> pd.DataFrame:
    return (
        all_eval
        .groupby("indicator", as_index=False)
        .agg(
            pairwise_accuracy_mean=("pairwise_accuracy", "mean"),
            pairwise_accuracy_std=("pairwise_accuracy", "std"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            kendall_mean=("kendall_tau_b", "mean"),
            kendall_std=("kendall_tau_b", "std"),
        )
    )


def company_feature_vector(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    feature: Dict[str, float] = {
        "paragraphs": float(len(records))
    }

    for field in FIELDS:
        counts = Counter(
            normalize_gold(item)[field]
            for item in records
        )
        for label in LABEL_SPACE[field]:
            feature[f"{field}::{label}"] = float(
                counts.get(label, 0)
            )

    return feature


def objective_score(
    fold_vectors: List[np.ndarray],
    fold_company_counts: List[int],
    total_vector: np.ndarray,
    n_splits: int,
) -> float:
    target = total_vector / n_splits
    scale = np.maximum(target, 1.0)

    matrix = np.vstack(fold_vectors)
    vector_loss = float(
        (((matrix - target) / scale) ** 2).mean()
    )

    company_target = sum(fold_company_counts) / n_splits
    company_scale = max(company_target, 1.0)

    company_loss = float(
        np.mean(
            [
                (
                    (count - company_target)
                    / company_scale
                ) ** 2
                for count in fold_company_counts
            ]
        )
    )

    return vector_loss + 0.20 * company_loss


def build_company_disjoint_folds(
    records: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    by_company = group_by_company(records)

    if len(by_company) < n_splits:
        raise ValueError(
            f"企業數 {len(by_company)} 小於 n_splits={n_splits}。"
        )

    features = {
        company: company_feature_vector(items)
        for company, items in by_company.items()
    }

    feature_names = sorted(
        next(iter(features.values())).keys()
    )

    vectors = {
        company: np.array(
            [features[company][name] for name in feature_names],
            dtype=float,
        )
        for company in by_company
    }

    total_vector = sum(
        vectors.values(),
        np.zeros(len(feature_names)),
    )

    rng = random.Random(seed)
    companies = list(by_company.keys())
    rng.shuffle(companies)

    target = total_vector / n_splits
    scale = np.maximum(target, 1.0)

    companies.sort(
        key=lambda company: (
            len(by_company[company]),
            float(np.linalg.norm(vectors[company] / scale)),
        ),
        reverse=True,
    )

    folds = [
        {
            "fold": fold_index,
            "companies": [],
            "records": [],
        }
        for fold_index in range(n_splits)
    ]

    fold_vectors = [
        np.zeros(len(feature_names), dtype=float)
        for _ in range(n_splits)
    ]
    fold_company_counts = [0 for _ in range(n_splits)]

    for company in companies:
        candidates = []

        for fold_index in range(n_splits):
            new_vectors = [
                vector.copy()
                for vector in fold_vectors
            ]
            new_counts = list(fold_company_counts)

            new_vectors[fold_index] += vectors[company]
            new_counts[fold_index] += 1

            score = objective_score(
                new_vectors,
                new_counts,
                total_vector,
                n_splits,
            )
            candidates.append((score, fold_index))

        _, best_fold = min(candidates)

        folds[best_fold]["companies"].append(company)
        folds[best_fold]["records"].extend(by_company[company])
        fold_vectors[best_fold] += vectors[company]
        fold_company_counts[best_fold] += 1

    summary_rows = []

    for fold_index, fold in enumerate(folds):
        summary_rows.append(
            {
                "fold": fold_index,
                "n_companies": len(fold["companies"]),
                "n_paragraphs": len(fold["records"]),
                "companies": "|".join(sorted(fold["companies"])),
            }
        )

    return folds, pd.DataFrame(summary_rows)


def save_fold_files(
    records: List[Dict[str, Any]],
    folds: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    for fold in folds:
        fold_index = fold["fold"]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_records = fold["records"]
        test_ids = {
            normalize_id(item["id"])
            for item in test_records
        }

        train_records = [
            item
            for item in records
            if normalize_id(item["id"]) not in test_ids
        ]

        train_companies = {
            normalize_id(item["ticker"])
            for item in train_records
        }
        test_companies = {
            normalize_id(item["ticker"])
            for item in test_records
        }

        overlap = train_companies & test_companies

        if overlap:
            raise RuntimeError(
                f"Fold {fold_index} 發生企業洩漏：{sorted(overlap)}"
            )

        save_json(train_records, fold_dir / "train.json")
        save_json(test_records, fold_dir / "test.json")
        save_json(
            {
                "fold": fold_index,
                "split_type": "company_disjoint",
                "train_companies": sorted(train_companies),
                "test_companies": sorted(test_companies),
                "train_samples": len(train_records),
                "test_samples": len(test_records),
            },
            fold_dir / "split_info.json",
        )


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(args.data)
    validate_dataset(records)

    folds, fold_summary = build_company_disjoint_folds(
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
        records,
        folds,
        args.output_dir,
    )

    print("=" * 90)
    print(
        "Company-disjoint 5-fold | "
        "Qwen Mean Company Embedding + Pairwise Ranking"
    )
    print("=" * 90)
    print(f"資料筆數：{len(records)}")
    print(f"企業數：{len(group_by_company(records))}")
    print(
        fold_summary[
            ["fold", "n_companies", "n_paragraphs"]
        ].to_string(index=False)
    )

    if args.prepare_only:
        return

    paragraph_embeddings = encode_all_paragraphs(
        records=records,
        model_name=args.embedding_model,
        max_length=args.max_length,
        batch_size=args.embedding_batch_size,
        cache_path=args.output_dir / "paragraph_embeddings.npz",
        recompute=args.recompute_embeddings,
    )

    all_company_rankings = []
    all_pair_predictions = []
    all_evaluations = []
    all_training_pairs = []

    for fold_index in range(args.n_splits):
        fold_dir = args.output_dir / f"fold_{fold_index}"

        train_records = load_json(fold_dir / "train.json")
        test_records = load_json(fold_dir / "test.json")

        train_df, train_x = build_company_table(
            train_records,
            paragraph_embeddings,
        )
        test_df, test_x = build_company_table(
            test_records,
            paragraph_embeddings,
        )

        train_companies = set(train_df["ticker"])
        test_companies = set(test_df["ticker"])
        overlap = train_companies & test_companies
        if overlap:
            raise RuntimeError(
                f"Fold {fold_index}: company leakage detected: {sorted(overlap)}"
            )

        print("\n" + "=" * 90)
        print(f"Fold {fold_index}")
        print(
            f"Train companies={len(train_df)}, "
            f"Test companies={len(test_df)}"
        )
        print("=" * 90)

        if len(train_df) < 2:
            raise RuntimeError(
                f"Fold {fold_index}: training companies 不足。"
            )
        if len(test_df) < 2:
            raise RuntimeError(
                f"Fold {fold_index}: test companies 少於 2，無法 pairwise ranking。"
            )

        fold_rankings = test_df.copy()
        fold_pair_dfs = []
        fold_eval_dfs = []

        for indicator in INDICATORS:
            pair_x, pair_y, train_pair_df = build_pairwise_training_data(
                train_df,
                train_x,
                indicator,
            )
            train_pair_df["fold"] = fold_index

            model = fit_pairwise_ranker(
                pair_x,
                pair_y,
                logistic_c=args.logistic_c,
                max_iter=args.max_iter,
                seed=args.seed,
            )

            ranked_df, pair_pred_df = rank_test_companies_allpair(
                test_df,
                test_x,
                indicator,
                model,
            )
            pair_pred_df["fold"] = fold_index

            eval_df = evaluate_pairwise_ranking(
                ranked_df,
                pair_pred_df,
                indicator,
                fold_index,
            )

            keep_cols = [
                "ticker",
                f"gold_{indicator}",
                f"wins_{indicator}",
                f"rank_score_{indicator}",
                f"pred_rank_{indicator}",
                f"gold_rank_{indicator}",
            ]
            fold_rankings = fold_rankings.merge(
                ranked_df[keep_cols],
                on="ticker",
                how="left",
                validate="one_to_one",
            )

            train_pair_df.to_csv(
                fold_dir / f"{indicator}_training_pairs.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pair_pred_df.to_csv(
                fold_dir / f"{indicator}_test_pair_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

            fold_pair_dfs.append(pair_pred_df)
            fold_eval_dfs.append(eval_df)
            all_training_pairs.append(train_pair_df)

        fold_rankings["fold"] = fold_index
        fold_pairs = pd.concat(fold_pair_dfs, ignore_index=True)
        fold_eval = pd.concat(fold_eval_dfs, ignore_index=True)

        fold_rankings.to_csv(
            fold_dir / "company_pairwise_rankings.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_pairs.to_csv(
            fold_dir / "all_test_pair_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fold_eval.to_csv(
            fold_dir / "pairwise_ranking_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_company_rankings.append(fold_rankings)
        all_pair_predictions.append(fold_pairs)
        all_evaluations.append(fold_eval)

    all_company = pd.concat(
        all_company_rankings,
        ignore_index=True,
    )
    all_pairs = pd.concat(
        all_pair_predictions,
        ignore_index=True,
    )
    all_eval = pd.concat(
        all_evaluations,
        ignore_index=True,
    )
    all_train_pairs = pd.concat(
        all_training_pairs,
        ignore_index=True,
    )

    all_company.to_csv(
        args.output_dir / "fold_company_pairwise_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_pairs.to_csv(
        args.output_dir / "fold_test_pair_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_train_pairs.to_csv(
        args.output_dir / "fold_training_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_eval.to_csv(
        args.output_dir / "fold_pairwise_ranking_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mean_std = build_mean_std(all_eval)
    mean_std.to_csv(
        args.output_dir / "pairwise_ranking_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPairwise ranking fold mean ± std：")
    print(
        mean_std.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\n注意：Company-disjoint 5-fold 不建立 global OOF ranking。"
        "每家公司雖只出現在一個 test fold，但不同 test folds 之間沒有直接 pairwise comparison；"
        "因此各 fold 的 win-rate / rank score 不能直接拼成全 49 家企業的共同排名。"
    )

    print(f"\n輸出資料夾：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
