#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random stratified 5-fold：Cross-Company Supervised Contrastive QLoRA

功能
----
1. 讀取合併後的 ESG JSON。
2. 對全部 2,000 筆段落建立 Random stratified 5-fold。
3. 同一家企業可同時出現在 train/test，作為 seen-company 對照設定。
4. 優先使用四個任務的複合標籤分層；罕見組合會自動回退到較粗粒度 strata。
5. 每 fold：
   - 訓練 QLoRA
   - 對隨機保留段落推論
   - 計算四任務 Macro-F1 / Accuracy / Balanced Accuracy / Per-class metrics
   - 聚合企業 ECR / CER / BHCR
   - 計算 MAE / RMSE / Pearson / Spearman
6. 合併所有 fold 的 OOF predictions。
7. 建立全資料 OOF 企業指標與全域評估結果。

依賴
----
pip install torch transformers peft bitsandbytes datasets accelerate \
            pandas numpy scipy scikit-learn tqdm

執行
----
python src/run_random_5fold_contrastive_qlora.py \
  --data data/vpesg4k_train_val.json \
  --output-dir runs/random_contrastive \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu 1 \
  --contrastive-lambda 0.10
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
# Set CUDA visibility from --gpu before importing torch.
import sys
_gpu = next((v.split("=", 1)[1] for v in sys.argv[1:] if v.startswith("--gpu=")), "0")
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

"""
python src/run_random_5fold.py \
  --data data/vpesg4k_train_val.json \
  --output-dir runs/random_contrastive \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --gpu 1 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 4 \
  --learning-rate 1e-4
"""

# =========================================================
# 1. 常數
# =========================================================

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

CER_DENOMINATOR = "evidence"

RQ1_WEIGHTS = {
    "promise_status": 0.20,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
    "verification_timeline": 0.15,
}

RQ1_TASK_NAMES = {
    "promise_status": "Commitment Classification",
    "evidence_status": "Evidence Identification",
    "evidence_quality": "Clarity Classification",
    "verification_timeline": "Timeline Classification",
}

SYSTEM_PROMPT = (
    "你是 ESG 永續承諾驗證分類器。"
    "你的任務是只根據輸入文本，判斷企業在 ESG 報告中是否提出可被追蹤與驗證的永續承諾，"
    "並判斷承諾是否有具體證據支持、證據品質，以及驗證時間軸。"
    "你只能輸出 JSON，不能加入任何解釋、Markdown 或多餘文字。\n\n"

    f"promise_status 合法標籤：{LABEL_SPACE['promise_status']}\n"
    f"verification_timeline 合法標籤：{LABEL_SPACE['verification_timeline']}\n"
    f"evidence_status 合法標籤：{LABEL_SPACE['evidence_status']}\n"
    f"evidence_quality 合法標籤：{LABEL_SPACE['evidence_quality']}\n\n"

    "promise_status：\n"
    "- Yes：包含企業對 ESG 行動、政策、制度、目標、改善方向或未來作為的明確承諾。\n"
    "- No：只有背景、成果、現況、產品或報告說明，沒有明確承諾。\n\n"

    "evidence_status：\n"
    "- Yes：有具體行動、制度、數據、計畫、政策、成果、查證或追蹤機制支持。\n"
    "- No：只有願景、口號、原則或抽象方向。\n"
    "- N/A：沒有承諾。\n\n"

    "evidence_quality：\n"
    "- Clear：證據具體、明確且直接支持承諾。\n"
    "- Not Clear：相關但籠統、缺乏量化、執行方式或成效。\n"
    "- Misleading：關聯弱、偏題或容易誤導。\n"
    "- N/A：沒有承諾，或 evidence_status = No。\n\n"

    "verification_timeline：\n"
    "- already：已完成、正在落實、定期或當年度可驗證。\n"
    "- within_2_years：兩年內可驗證。\n"
    "- between_2_and_5_years：二至五年內，或一般中期規劃。\n"
    "- more_than_5_years：超過五年，例如 2030、2035、2050 或長期目標。\n"
    "- N/A：沒有承諾。\n\n"

    "一致性規則：\n"
    "1. promise_status = No 時，其餘三欄必須是 N/A。\n"
    "2. promise_status = Yes 且 evidence_status = No 時，evidence_quality 必須是 N/A。\n"
    "3. 嚴格使用合法標籤與固定 JSON 格式。"
)


# =========================================================
# 2. 基礎工具
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive-lambda", type=float, default=0.10)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-pairs-per-step", type=int, default=2)
    parser.add_argument("--contrastive-max-length", type=int, default=512)
    parser.add_argument(
        "--contrastive-fields",
        nargs="+",
        default=FIELDS,
        choices=FIELDS,
        help="輪流用哪些任務標籤建立 supervised contrastive batch。",
    )

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="已存在 predictions.json 的 fold 會跳過訓練與推論。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    if value == "":
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


# =========================================================
# 3. Random stratified fold 建立
# =========================================================

def make_stratification_labels(
    records: List[Dict[str, Any]],
    n_splits: int,
) -> Tuple[List[str], str]:
    """
    建立 Random 5-fold 的分層標籤。

    優先順序：
    1. 四任務完整複合標籤
    2. Promise + Evidence + Quality
    3. Promise + Evidence
    4. Promise

    StratifiedKFold 要求每個 stratum 至少有 n_splits 筆。
    若完整複合標籤包含罕見組合，會自動回退到較粗粒度標籤。
    """
    gold_rows = [normalize_gold(item) for item in records]

    candidates = [
        (
            "promise+timeline+evidence+quality",
            [
                "||".join(
                    [
                        gold["promise_status"],
                        gold["verification_timeline"],
                        gold["evidence_status"],
                        gold["evidence_quality"],
                    ]
                )
                for gold in gold_rows
            ],
        ),
        (
            "promise+evidence+quality",
            [
                "||".join(
                    [
                        gold["promise_status"],
                        gold["evidence_status"],
                        gold["evidence_quality"],
                    ]
                )
                for gold in gold_rows
            ],
        ),
        (
            "promise+evidence",
            [
                "||".join(
                    [
                        gold["promise_status"],
                        gold["evidence_status"],
                    ]
                )
                for gold in gold_rows
            ],
        ),
        (
            "promise",
            [gold["promise_status"] for gold in gold_rows],
        ),
    ]

    for name, labels in candidates:
        counts = Counter(labels)
        if min(counts.values()) >= n_splits:
            return labels, name

    raise ValueError(
        "即使只使用 promise_status，也存在少於 n_splits 的類別，"
        "無法建立 StratifiedKFold。"
    )


def build_random_folds(
    records: List[Dict[str, Any]],
    n_splits: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, str]:
    labels, strategy = make_stratification_labels(records, n_splits)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    indices = np.arange(len(records))
    folds: List[Dict[str, Any]] = []
    summary_rows = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(indices, labels)
    ):
        train_records = [records[int(i)] for i in train_indices]
        test_records = [records[int(i)] for i in test_indices]

        train_companies = {
            normalize_id(item["ticker"])
            for item in train_records
        }
        test_companies = {
            normalize_id(item["ticker"])
            for item in test_records
        }
        overlap = train_companies & test_companies

        folds.append(
            {
                "fold": fold_index,
                "train_records": train_records,
                "test_records": test_records,
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
            }
        )

        row: Dict[str, Any] = {
            "fold": fold_index,
            "n_train": len(train_records),
            "n_test": len(test_records),
            "n_train_companies": len(train_companies),
            "n_test_companies": len(test_companies),
            "n_overlapping_companies": len(overlap),
            "stratification_strategy": strategy,
        }

        # 報告各 fold 測試集四任務分布
        for field in FIELDS:
            counts = Counter(
                normalize_gold(item)[field]
                for item in test_records
            )
            for label in LABEL_SPACE[field]:
                row[f"test::{field}::{label}"] = int(
                    counts.get(label, 0)
                )

        summary_rows.append(row)

    return folds, pd.DataFrame(summary_rows), strategy


def save_fold_files(
    records: List[Dict[str, Any]],
    folds: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    all_ids = {normalize_id(item["id"]) for item in records}

    for fold in folds:
        fold_index = fold["fold"]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = fold["train_records"]
        test_records = fold["test_records"]

        train_ids = {normalize_id(item["id"]) for item in train_records}
        test_ids = {normalize_id(item["id"]) for item in test_records}

        if train_ids & test_ids:
            raise RuntimeError(
                f"Fold {fold_index} 發生樣本 ID 洩漏。"
            )

        if train_ids | test_ids != all_ids:
            raise RuntimeError(
                f"Fold {fold_index} 的 train/test 未完整覆蓋資料。"
            )

        train_companies = {
            normalize_id(item["ticker"])
            for item in train_records
        }
        test_companies = {
            normalize_id(item["ticker"])
            for item in test_records
        }

        save_json(train_records, fold_dir / "train.json")
        save_json(test_records, fold_dir / "test.json")
        save_json(
            {
                "fold": fold_index,
                "split_type": "random_stratified",
                "train_samples": len(train_records),
                "test_samples": len(test_records),
                "train_companies": sorted(train_companies),
                "test_companies": sorted(test_companies),
                "overlapping_companies": sorted(
                    train_companies & test_companies
                ),
                "train_indices": fold["train_indices"],
                "test_indices": fold["test_indices"],
            },
            fold_dir / "split_info.json",
        )


# =========================================================
# 4. Prompt 與 tokenization
# =========================================================

def build_input_text(item: Dict[str, Any]) -> str:
    return f"文本：\n{str(item.get('data', '')).strip()}"


def build_gold_output(item: Dict[str, Any]) -> str:
    gold = normalize_gold(item)
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))


def build_training_prompt(item: Dict[str, Any]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_input_text(item)}<|im_end|>\n"
        f"<|im_start|>assistant\n{build_gold_output(item)}<|im_end|>\n"
    )


def build_inference_prompt(item: Dict[str, Any]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_input_text(item)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def create_tokenizer(base_model: str, padding_side: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side
    return tokenizer


def build_preprocess_fn(tokenizer: Any, max_length: int):
    assistant_ids = tokenizer(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = build_training_prompt(example)

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = tokenized["input_ids"]
        labels = [-100] * len(input_ids)

        start_index = -1
        for i in range(len(input_ids) - len(assistant_ids) + 1):
            if input_ids[i:i + len(assistant_ids)] == assistant_ids:
                start_index = i + len(assistant_ids)
                break

        if start_index < 0:
            raise ValueError("找不到 assistant 區塊。")

        for i in range(start_index, len(input_ids)):
            labels[i] = input_ids[i]

        tokenized["labels"] = labels
        return tokenized

    return preprocess


# =========================================================
# 5. 模型訓練與推論
# =========================================================

class CrossDomainContrastiveSampler:
    """
    每個 contrastive step 針對一個任務 field：
    1. 選擇至少出現在兩個 domain 的 labels；
    2. 對每個選中 label，抽兩筆相同 label、不同 domain 的樣本；
    3. 形成 2 * pairs_per_step 筆 contrastive mini-batch。

    Company setting 的 domain = ticker；Industry setting 的 domain = industry。
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        fields: List[str],
        domain_key: str,
        seed: int,
    ) -> None:
        self.records = records
        self.fields = list(fields)
        self.domain_key = domain_key
        self.rng = random.Random(seed)
        self.step = 0

        self.index: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
        for field in self.fields:
            field_map: Dict[str, Dict[str, List[int]]] = {}
            for idx, item in enumerate(records):
                label = normalize_gold(item)[field]
                domain = str(item.get(domain_key, "")).strip()
                if not domain:
                    raise ValueError(
                        f"Contrastive sampling 缺少 domain '{domain_key}'："
                        f"id={item.get('id')}"
                    )
                field_map.setdefault(label, {}).setdefault(domain, []).append(idx)
            self.index[field] = field_map

        self.valid_labels: Dict[str, List[str]] = {}
        for field in self.fields:
            self.valid_labels[field] = [
                label
                for label, domain_map in self.index[field].items()
                if len(domain_map) >= 2
            ]
            if not self.valid_labels[field]:
                raise ValueError(
                    f"field={field} 沒有可建立跨 domain positive pair 的 label。"
                )

    def sample(self, pairs_per_step: int) -> Tuple[str, List[int], List[str]]:
        field = self.fields[self.step % len(self.fields)]
        self.step += 1

        labels = list(self.valid_labels[field])
        self.rng.shuffle(labels)

        # 優先用不同 labels，讓 batch 內有明確 negatives。
        if len(labels) >= pairs_per_step:
            chosen_labels = labels[:pairs_per_step]
        else:
            chosen_labels = [
                labels[i % len(labels)] for i in range(pairs_per_step)
            ]

        indices: List[int] = []
        batch_labels: List[str] = []

        for label in chosen_labels:
            domain_map = self.index[field][label]
            domains = list(domain_map.keys())
            d1, d2 = self.rng.sample(domains, 2)
            i = self.rng.choice(domain_map[d1])
            j = self.rng.choice(domain_map[d2])
            indices.extend([i, j])
            batch_labels.extend([label, label])

        return field, indices, batch_labels


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: List[str],
    temperature: float,
) -> torch.Tensor:
    """標準 supervised contrastive objective（單 view，每個 anchor 可有多個 positives）。"""
    if features.ndim != 2:
        raise ValueError("features 必須是 [batch, hidden]。")

    z = F.normalize(features, p=2, dim=-1)
    logits = torch.matmul(z, z.T) / temperature

    n = logits.size(0)
    device = logits.device
    eye = torch.eye(n, dtype=torch.bool, device=device)

    label_ids: Dict[str, int] = {}
    numeric = []
    for label in labels:
        if label not in label_ids:
            label_ids[label] = len(label_ids)
        numeric.append(label_ids[label])
    y = torch.tensor(numeric, device=device)

    positive_mask = y.unsqueeze(0).eq(y.unsqueeze(1)) & (~eye)
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return logits.sum() * 0.0

    # 排除 self-similarity，並用 logsumexp 做穩定 normalization。
    masked_logits = logits.masked_fill(eye, float("-inf"))
    log_den = torch.logsumexp(masked_logits, dim=1)

    # 每個 anchor 對所有 positives 的 log-probability 取平均。
    log_prob = logits - log_den.unsqueeze(1)
    pos_count = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (
        log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / pos_count
    )

    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


class CrossDomainContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        raw_train_records: List[Dict[str, Any]],
        contrastive_fields: List[str],
        contrastive_domain_key: str,
        contrastive_lambda: float,
        contrastive_temperature: float,
        contrastive_pairs_per_step: int,
        contrastive_max_length: int,
        contrastive_tokenizer: Any,
        contrastive_seed: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raw_train_records = raw_train_records
        self.contrastive_lambda = float(contrastive_lambda)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_pairs_per_step = int(contrastive_pairs_per_step)
        self.contrastive_max_length = int(contrastive_max_length)
        self.contrastive_tokenizer = contrastive_tokenizer
        self.contrastive_sampler = CrossDomainContrastiveSampler(
            records=raw_train_records,
            fields=contrastive_fields,
            domain_key=contrastive_domain_key,
            seed=contrastive_seed,
        )

    def _encode_contrastive_batch(
        self,
        model: Any,
        records: List[Dict[str, Any]],
    ) -> torch.Tensor:
        texts = [build_input_text(item) for item in records]
        tokenized = self.contrastive_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.contrastive_max_length,
            return_tensors="pt",
        )

        input_device = model.get_input_embeddings().weight.device
        tokenized = {
            key: value.to(input_device)
            for key, value in tokenized.items()
        }

        outputs = model(
            **tokenized,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = tokenized["attention_mask"]

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)
        task_loss = outputs.loss

        if self.contrastive_lambda <= 0.0:
            return (task_loss, outputs) if return_outputs else task_loss

        field, sampled_indices, sampled_labels = self.contrastive_sampler.sample(
            self.contrastive_pairs_per_step
        )
        sampled_records = [
            self.raw_train_records[idx] for idx in sampled_indices
        ]

        features = self._encode_contrastive_batch(model, sampled_records)
        contrastive_loss = supervised_contrastive_loss(
            features=features,
            labels=sampled_labels,
            temperature=self.contrastive_temperature,
        )

        loss = task_loss + self.contrastive_lambda * contrastive_loss

        # 每隔一段時間將 auxiliary loss 寫入 Trainer log history。
        if self.state.global_step % 25 == 0:
            self.log({
                "task_loss_aux": float(task_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "contrastive_field": field,
            })

        return (loss, outputs) if return_outputs else loss

def create_quantized_base_model(
    base_model: str,
    use_cache: bool,
) -> Any:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = use_cache
    return model


def train_fold(
    train_records: List[Dict[str, Any]],
    adapter_dir: Path,
    args: argparse.Namespace,
) -> None:
    tokenizer = create_tokenizer(args.base_model, padding_side="right")
    base = create_quantized_base_model(args.base_model, use_cache=False)

    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base, lora_config)

    dataset = Dataset.from_list(train_records)
    preprocess_fn = build_preprocess_fn(tokenizer, args.max_length)

    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    print(
        "Contrastive config: "
        f"lambda={args.contrastive_lambda}, "
        f"temperature={args.contrastive_temperature}, "
        f"pairs_per_step={args.contrastive_pairs_per_step}, "
        f"fields={args.contrastive_fields}, "
        f"domain=ticker"
    )

    trainer = CrossDomainContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        raw_train_records=train_records,
        contrastive_fields=args.contrastive_fields,
        contrastive_domain_key="ticker",
        contrastive_lambda=args.contrastive_lambda,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_pairs_per_step=args.contrastive_pairs_per_step,
        contrastive_max_length=args.contrastive_max_length,
        contrastive_tokenizer=tokenizer,
        contrastive_seed=args.seed,
    )

    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model, base, tokenizer, dataset, tokenized_dataset
    gc.collect()
    torch.cuda.empty_cache()


def extract_json_text(text: str) -> str:
    text = text.replace("<|im_end|>", "").strip()
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def apply_prediction_consistency(pred: Dict[str, str]) -> Dict[str, str]:
    if pred["promise_status"] == "No":
        pred["verification_timeline"] = "N/A"
        pred["evidence_status"] = "N/A"
        pred["evidence_quality"] = "N/A"

    if pred["evidence_status"] in ["No", "N/A"]:
        pred["evidence_quality"] = "N/A"

    return pred


def parse_prediction(generated_text: str) -> Dict[str, str]:
    try:
        obj = json.loads(extract_json_text(generated_text))
    except Exception:
        obj = {}

    pred: Dict[str, str] = {}

    for field in FIELDS:
        value = str(obj.get(field, "N/A")).strip() or "N/A"

        if (
            field == "verification_timeline"
            and value == "longer_than_5_years"
        ):
            value = "more_than_5_years"

        if value not in LABEL_SPACE[field]:
            compact = value.lower().replace(" ", "")
            matched = next(
                (
                    legal
                    for legal in LABEL_SPACE[field]
                    if compact == legal.lower().replace(" ", "")
                ),
                None,
            )
            value = matched if matched is not None else "N/A"

        pred[field] = value

    return apply_prediction_consistency(pred)


@torch.no_grad()
def predict_fold(
    test_records: List[Dict[str, Any]],
    adapter_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    tokenizer = create_tokenizer(args.base_model, padding_side="left")
    base = create_quantized_base_model(args.base_model, use_cache=True)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    predictions: List[Dict[str, Any]] = []

    for item in tqdm(test_records, desc=f"Predicting {adapter_dir.parent.name}"):
        prompt = build_inference_prompt(item)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        ).strip()

        pred = parse_prediction(generated_text)
        gold = normalize_gold(item)

        row = {
            "id": normalize_id(item["id"]),
            "company": item.get("company", ""),
            "ticker": normalize_id(item["ticker"]),
            "data": item.get("data", ""),
            "raw_generation": generated_text,
            "prediction": pred,
            "gold": gold,
        }

        for field in FIELDS:
            row[f"pred_{field}"] = pred[field]
            row[f"gold_{field}"] = gold[field]

        predictions.append(row)

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


# =========================================================
# 6. 段落級評估
# =========================================================

def classification_metrics(
    prediction_rows: List[Dict[str, Any]],
    fold: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []

    for field in FIELDS:
        y_true = [row[f"gold_{field}"] for row in prediction_rows]
        y_pred = [row[f"pred_{field}"] for row in prediction_rows]
        labels = LABEL_SPACE[field]

        summary_rows.append(
            {
                "fold": fold,
                "field": field,
                "n_samples": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        for label, p, r, f, s in zip(
            labels, precision, recall, f1, support
        ):
            class_rows.append(
                {
                    "fold": fold,
                    "field": field,
                    "label": label,
                    "support": int(s),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows)


# =========================================================
# 7. 企業指標
# =========================================================

def safe_ratio(numerator: int, denominator: int) -> float:
    return np.nan if denominator == 0 else numerator / denominator


def aggregate_company_metrics(
    prediction_rows: List[Dict[str, Any]],
    prefix: str,
) -> pd.DataFrame:
    """
    聚合 paragraph-level predictions / gold labels 成企業級指標。

    指標定義
    --------
    ECR =
        N(Promise=Yes AND Evidence=Yes)
        / N(Promise=Yes)

    CER =
        N(Promise=Yes AND Evidence=Yes AND Quality=Clear)
        / N(Promise=Yes AND Evidence=Yes)

    SHCR =
        [N(already) + N(within_2_years)]
        / N(Promise=Yes)

    LHCR =
        N(more_than_5_years)
        / N(Promise=Yes)

    BHCR =
        2 * SHCR * LHCR / (SHCR + LHCR)

    若 SHCR = 0 且 LHCR = 0，則 BHCR = 0。

    注意：
    between_2_and_5_years 屬於中期承諾，
    不列入 SHCR 或 LHCR 的分子，
    但若 promise_status = Yes，仍包含在 Promise 分母中。
    """
    df = pd.DataFrame(prediction_rows)
    rows = []

    for ticker, group in df.groupby("ticker", sort=True):
        promise = group[f"{prefix}_promise_status"]
        timeline = group[f"{prefix}_verification_timeline"]
        evidence = group[f"{prefix}_evidence_status"]
        quality = group[f"{prefix}_evidence_quality"]

        # -------------------------------------------------
        # 基礎計數
        # -------------------------------------------------
        promises = int(
            promise.eq("Yes").sum()
        )

        evidence_yes = int(
            (
                promise.eq("Yes")
                & evidence.eq("Yes")
            ).sum()
        )

        clear = int(
            (
                promise.eq("Yes")
                & evidence.eq("Yes")
                & quality.eq("Clear")
            ).sum()
        )

        # 短期：already + within_2_years
        short_horizon = int(
            (
                promise.eq("Yes")
                & timeline.isin(
                    [
                        "already",
                        "within_2_years",
                    ]
                )
            ).sum()
        )

        # 中期：只作為診斷計數，不進 SHCR / LHCR 分子
        medium_horizon = int(
            (
                promise.eq("Yes")
                & timeline.eq(
                    "between_2_and_5_years"
                )
            ).sum()
        )

        # 長期：more_than_5_years
        long_horizon = int(
            (
                promise.eq("Yes")
                & timeline.eq(
                    "more_than_5_years"
                )
            ).sum()
        )

        cer_denominator = (
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
            cer_denominator,
        )

        shcr = safe_ratio(
            short_horizon,
            promises,
        )
        lhcr = safe_ratio(
            long_horizon,
            promises,
        )

        # -------------------------------------------------
        # BHCR：SHCR 與 LHCR 的 harmonic mean
        # -------------------------------------------------
        if pd.isna(shcr) or pd.isna(lhcr):
            bhcr = np.nan
        elif shcr == 0.0 and lhcr == 0.0:
            bhcr = 0.0
        else:
            denominator = shcr + lhcr
            bhcr = (
                0.0
                if denominator == 0.0
                else (
                    2.0
                    * shcr
                    * lhcr
                    / denominator
                )
            )

        rows.append(
            {
                "ticker": ticker,
                "company": group["company"].iloc[0],
                f"{prefix}_paragraphs": len(group),
                f"{prefix}_promises": promises,
                f"{prefix}_evidence_yes": evidence_yes,
                f"{prefix}_clear": clear,

                # 保留時間組成計數，方便檢查 BHCR
                f"{prefix}_short_horizon": short_horizon,
                f"{prefix}_medium_horizon": medium_horizon,
                f"{prefix}_long_horizon": long_horizon,

                f"{prefix}_ECR": ecr,
                f"{prefix}_CER": cer,

                # SHCR / LHCR 為 BHCR 的中間量，保留在輸出供 audit
                f"{prefix}_SHCR": shcr,
                f"{prefix}_LHCR_component": lhcr,

                # 正式第三指標
                f"{prefix}_BHCR": bhcr,
            }
        )

    return pd.DataFrame(rows)


def safe_corr(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> Tuple[float, float, int]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(valid) < 3:
        return np.nan, np.nan, len(valid)

    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan, len(valid)

    result = (
        pearsonr(valid["x"], valid["y"])
        if method == "pearson"
        else spearmanr(valid["x"], valid["y"])
    )

    return float(result.statistic), float(result.pvalue), len(valid)


def evaluate_company_metrics(
    company_df: pd.DataFrame,
    fold: Any,
) -> pd.DataFrame:
    rows = []

    for indicator in ["ECR", "CER", "BHCR"]:
        true_col = f"gold_{indicator}"
        pred_col = f"pred_{indicator}"
        valid = company_df[[true_col, pred_col]].dropna()

        if len(valid):
            error = valid[pred_col] - valid[true_col]
            mae = float(np.abs(error).mean())
            rmse = float(np.sqrt(np.square(error).mean()))
        else:
            mae = np.nan
            rmse = np.nan

        spearman_rho, spearman_p, spearman_n = safe_corr(
            company_df[true_col],
            company_df[pred_col],
            "spearman",
        )

        if (
            len(valid) >= 2
            and valid[true_col].nunique() >= 2
            and valid[pred_col].nunique() >= 2
        ):
            kt = kendalltau(
                valid[true_col],
                valid[pred_col],
                variant="b",
                nan_policy="omit",
            )
            kendall_tau_b = float(kt.statistic)
            kendall_p = float(kt.pvalue)
        else:
            kendall_tau_b = np.nan
            kendall_p = np.nan

        rows.append({
            "fold": fold,
            "indicator": indicator,
            "n_companies": len(valid),
            "mae": mae,
            "rmse": rmse,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "kendall_tau_b": kendall_tau_b,
            "kendall_p": kendall_p,
            "n_spearman": spearman_n,
        })

    return pd.DataFrame(rows)


def build_rq1_results(
    all_classification_df: pd.DataFrame,
    oof_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for fold in sorted(all_classification_df["fold"].unique()):
        fold_df = all_classification_df[all_classification_df["fold"] == fold]
        score_map = dict(zip(fold_df["field"], fold_df["macro_f1"]))
        row = {"scope": "fold", "fold": fold}
        weighted = 0.0
        for field in FIELDS:
            value = float(score_map[field])
            row[f"{field}_macro_f1"] = value
            weighted += RQ1_WEIGHTS[field] * value
        row["weighted_macro_f1"] = weighted
        rows.append(row)

    fold_df_out = pd.DataFrame(rows)
    for stat_name, func in [("mean", "mean"), ("std", "std")]:
        row = {"scope": f"5fold_{stat_name}", "fold": ""}
        for field in FIELDS:
            row[f"{field}_macro_f1"] = getattr(fold_df_out[f"{field}_macro_f1"], func)()
        row["weighted_macro_f1"] = getattr(fold_df_out["weighted_macro_f1"], func)()
        rows.append(row)

    oof_map = dict(zip(oof_summary["field"], oof_summary["macro_f1"]))
    row = {"scope": "OOF_global", "fold": ""}
    weighted = 0.0
    for field in FIELDS:
        value = float(oof_map[field])
        row[f"{field}_macro_f1"] = value
        weighted += RQ1_WEIGHTS[field] * value
    row["weighted_macro_f1"] = weighted
    rows.append(row)

    return pd.DataFrame(rows)[[
        "scope", "fold",
        "promise_status_macro_f1",
        "evidence_status_macro_f1",
        "evidence_quality_macro_f1",
        "verification_timeline_macro_f1",
        "weighted_macro_f1",
    ]]


def build_rq2_results(
    all_company_eval_df: pd.DataFrame,
    oof_company_eval: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, r in all_company_eval_df.iterrows():
        rows.append({
            "scope": "fold",
            "fold": r["fold"],
            "indicator": r["indicator"],
            "n_companies": r["n_companies"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "spearman_rho": r["spearman_rho"],
            "spearman_p": r["spearman_p"],
            "kendall_tau_b": r["kendall_tau_b"],
            "kendall_p": r["kendall_p"],
        })

    metrics = ["mae", "rmse", "spearman_rho", "kendall_tau_b"]
    for indicator, group in all_company_eval_df.groupby("indicator"):
        for stat_name, func in [("mean", "mean"), ("std", "std")]:
            row = {
                "scope": f"5fold_{stat_name}",
                "fold": "",
                "indicator": indicator,
                "n_companies": float(getattr(group["n_companies"], func)()),
                "spearman_p": np.nan,
                "kendall_p": np.nan,
            }
            for metric in metrics:
                row[metric] = float(getattr(group[metric], func)())
            rows.append(row)

    for _, r in oof_company_eval.iterrows():
        rows.append({
            "scope": "OOF_global",
            "fold": "",
            "indicator": r["indicator"],
            "n_companies": r["n_companies"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "spearman_rho": r["spearman_rho"],
            "spearman_p": r["spearman_p"],
            "kendall_tau_b": r["kendall_tau_b"],
            "kendall_p": r["kendall_p"],
        })

    return pd.DataFrame(rows)[[
        "scope", "fold", "indicator", "n_companies",
        "mae", "rmse", "spearman_rho", "spearman_p",
        "kendall_tau_b", "kendall_p",
    ]]

def build_company_result(
    prediction_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    gold = aggregate_company_metrics(prediction_rows, "gold")
    pred = aggregate_company_metrics(prediction_rows, "pred")

    merged = gold.merge(
        pred,
        on=["ticker", "company"],
        how="inner",
        validate="one_to_one",
    )

    for indicator in ["ECR", "CER", "BHCR"]:
        merged[f"{indicator}_error"] = (
            merged[f"pred_{indicator}"]
            - merged[f"gold_{indicator}"]
        )
        merged[f"{indicator}_absolute_error"] = (
            merged[f"{indicator}_error"].abs()
        )

    return merged


# =========================================================
# 8. 主流程
# =========================================================

def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(args.data)
    validate_dataset(records)

    print("=" * 80)
    print("Random stratified 5-fold + Cross-Company Supervised Contrastive QLoRA")
    print("=" * 80)
    print(f"資料筆數：{len(records)}")
    print(
        f"企業數：{len({normalize_id(item['ticker']) for item in records})}"
    )

    folds, fold_summary, stratification_strategy = build_random_folds(
        records=records,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    fold_summary.to_csv(
        args.output_dir / "fold_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_fold_files(records, folds, args.output_dir)

    print("\nFold 分布：")
    print(
        fold_summary[
            [
                "fold",
                "n_train",
                "n_test",
                "n_train_companies",
                "n_test_companies",
                "n_overlapping_companies",
            ]
        ].to_string(index=False)
    )

    if args.prepare_only:
        print("\n已建立 Random stratified folds；因 --prepare-only，不執行訓練。")
        return

    all_predictions: List[Dict[str, Any]] = []
    fold_classification = []
    fold_per_class = []
    fold_company_evaluation = []

    for fold_index in range(args.n_splits):
        fold_dir = args.output_dir / f"fold_{fold_index}"
        adapter_dir = fold_dir / "adapter"
        prediction_path = fold_dir / "predictions.json"

        train_records = load_json(fold_dir / "train.json")
        test_records = load_json(fold_dir / "test.json")

        print("\n" + "=" * 80)
        print(f"Fold {fold_index}")
        print(
            f"Train={len(train_records)}, Test={len(test_records)}"
        )
        print("=" * 80)

        if args.resume and prediction_path.exists():
            print("使用既有 predictions.json，跳過訓練與推論。")
            predictions = load_json(prediction_path)
        else:
            train_fold(
                train_records=train_records,
                adapter_dir=adapter_dir,
                args=args,
            )

            predictions = predict_fold(
                test_records=test_records,
                adapter_dir=adapter_dir,
                args=args,
            )
            save_json(predictions, prediction_path)

        for row in predictions:
            row["fold"] = fold_index

        all_predictions.extend(predictions)

        summary_df, class_df = classification_metrics(
            predictions,
            fold=fold_index,
        )
        company_df = build_company_result(predictions)
        company_eval_df = evaluate_company_metrics(
            company_df,
            fold=fold_index,
        )

        summary_df.to_csv(
            fold_dir / "classification_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        class_df.to_csv(
            fold_dir / "per_class_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        company_df.to_csv(
            fold_dir / "company_level_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        company_eval_df.to_csv(
            fold_dir / "company_indicator_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fold_classification.append(summary_df)
        fold_per_class.append(class_df)
        fold_company_evaluation.append(company_eval_df)

    # -----------------------------------------------------
    # OOF 驗證
    # -----------------------------------------------------
    if len(all_predictions) != len(records):
        raise RuntimeError(
            f"OOF 預測數 {len(all_predictions)} != 原始資料數 {len(records)}"
        )

    oof_ids = [normalize_id(row["id"]) for row in all_predictions]
    if len(oof_ids) != len(set(oof_ids)):
        raise RuntimeError("OOF predictions 出現重複 id。")

    save_json(
        all_predictions,
        args.output_dir / "oof_predictions.json",
    )

    all_classification_df = pd.concat(
        fold_classification,
        ignore_index=True,
    )
    all_per_class_df = pd.concat(
        fold_per_class,
        ignore_index=True,
    )
    all_company_eval_df = pd.concat(
        fold_company_evaluation,
        ignore_index=True,
    )

    all_classification_df.to_csv(
        args.output_dir / "fold_classification_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_per_class_df.to_csv(
        args.output_dir / "fold_per_class_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_company_eval_df.to_csv(
        args.output_dir / "fold_company_indicator_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Fold mean ± std
    classification_mean_std = (
        all_classification_df
        .groupby("field", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
    )
    classification_mean_std.to_csv(
        args.output_dir / "classification_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    company_mean_std = (
        all_company_eval_df
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
    company_mean_std.to_csv(
        args.output_dir / "company_indicator_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # OOF 全資料分類評估
    oof_summary, oof_per_class = classification_metrics(
        all_predictions,
        fold="OOF_global",
    )
    oof_summary.to_csv(
        args.output_dir / "oof_classification_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof_per_class.to_csv(
        args.output_dir / "oof_per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # OOF 全企業指標
    oof_company = build_company_result(all_predictions)
    oof_company_eval = evaluate_company_metrics(
        oof_company,
        fold="OOF_global",
    )

    oof_company.to_csv(
        args.output_dir / "oof_company_level_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    oof_company_eval.to_csv(
        args.output_dir / "oof_company_indicator_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rq1_results = build_rq1_results(
        all_classification_df=all_classification_df,
        oof_summary=oof_summary,
    )
    rq1_results.to_csv(
        args.output_dir / "RQ1_cross_company_generalization.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rq2_results = build_rq2_results(
        all_company_eval_df=all_company_eval_df,
        oof_company_eval=oof_company_eval,
    )
    rq2_results.to_csv(
        args.output_dir / "RQ2_inter_company_relationship.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("5-fold 完成")
    print("=" * 80)

    print("\n段落級 mean ± std：")
    print(
        classification_mean_std.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n企業指標 fold mean ± std：")
    print(
        company_mean_std.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOOF 全企業結果：")
    print(
        oof_company_eval.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n論文用 RQ 輸出：")
    print(f"RQ1: {(args.output_dir / 'RQ1_cross_company_generalization.csv').resolve()}")
    print(f"RQ2: {(args.output_dir / 'RQ2_inter_company_relationship.csv').resolve()}")
    print(f"\n輸出資料夾：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
