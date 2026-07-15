"""
stage02_preprocess.py — 数据预处理管道
=====================================
输入：stage01 输出的 DataFrame（data/data_stage01.csv，9374 × 40）
输出：预处理后的 DataFrame（data/data_stage02.csv，9374 × 78）

保留步骤：
    2. bool / 性别 → 0/1
    3. 计算 19 个症状得分列（×k=2.0）
    4. Z-score 标准化
    5. 症状得分 × k 放大
    6. 标签 one-hot 编码
    列重排：特征列在前，标签列在最后

依赖：
    - ml.stage01_load
    - ml.diagnostic_rules（提供 19 种疾病名和评分函数）

用法：
    from ml.stage01_load import load_and_prepare_data
    from ml.stage02_preprocess import preprocess

    df_raw = load_and_prepare_data()
    df_clean = preprocess(df_raw)
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 常量（硬编码，与 ml.diagnostic_rules 保持一致）
# ============================================================

# 与 ml.diagnostic_rules.get_diagnostic_rules_vec() 返回的 key 完全一致
TARGET_DISEASES: List[str] = [
    "高血压", "低血压", "睡眠呼吸暂停", "心律失常", "高尿酸血症",
    "糖尿病", "脂肪肝", "NAFLD", "痛风", "骨质疏松",
    "贫血", "感冒", "肠胃炎", "胆结石", "肾结石",
    "肾衰竭", "乙肝", "丙肝", "蛀牙",
]

LABEL_PREFIX: str = "患病情况_"
GENDER_MAP: Dict[str, int] = {"男": 1, "女": 0}
SCORE_PREFIX: str = "症状得分_"
DEFAULT_SCORE_K: float = 2.0


# ============================================================
# 步骤 2：bool / 性别 → 0/1
# ============================================================

def encode_bool_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    将所有 bool 型列（True/False）转为 1/0（int64）。
    列名从 ml.stage01_load.BOOL_COLS 读取。
    """
    from ml.stage01_load import BOOL_COLS

    df = df.copy()
    for col in BOOL_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == "bool":
            df[col] = df[col].astype("int64")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0})
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """性别：男→1, 女→0"""
    df = df.copy()
    col = "性别"
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].map(GENDER_MAP).astype("int64")
    return df


# ============================================================
# 步骤 3：计算症状得分（向量化）
# ============================================================

def compute_disease_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每行样本按 diagnostic_rules 中定义的 19 种疾病规则打分。
    结果追加在 df 末尾，列名如 "症状得分_高血压"。

    向量化实现（整列操作），速度约为逐行 apply 的 ~100 倍。
    """
    from ml.diagnostic_rules import get_diagnostic_rules_vec

    rules = get_diagnostic_rules_vec()
    logger.info(f"  [症状得分] 对 {len(rules)} 种疾病计算得分…")

    for disease, score_fn in rules.items():
        col = f"{SCORE_PREFIX}{disease}"
        df[col] = score_fn(df).astype("int64")

    return df


# ============================================================
# 步骤 4：Z-score 标准化（除症状得分 + 标签列）
# ============================================================

def zscore_normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    对所有原始特征列做 Z-score 标准化：(x - mean) / std。
    排除：
        - 原始标签列"患病情况"（字符串）
        - 症状得分列（SCORE_PREFIX 开头）
        - one-hot 标签列（LABEL_PREFIX 开头）

    返回:
        (标准化后的 DataFrame, 每列的 {mean, std} 字典)
    """
    df = df.copy()

    exclude = {"患病情况"}
    exclude |= {c for c in df.columns if c.startswith(SCORE_PREFIX)}
    exclude |= {c for c in df.columns if c.startswith(LABEL_PREFIX)}
    feature_cols = [c for c in df.columns if c not in exclude]

    logger.info(f"  [Z-score] 对 {len(feature_cols)} 列做标准化…")
    norm_stats: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=0)
        if std_val == 0 or pd.isna(std_val):
            logger.warning(f"  [Z-score] {col} 标准差为 0，跳过")
            norm_stats[col] = {"mean": float(mean_val), "std": 1.0}
            continue
        norm_stats[col] = {"mean": float(mean_val), "std": float(std_val)}
        df[col] = (df[col] - mean_val) / std_val

    return df, norm_stats


# ============================================================
# 步骤 5：症状得分乘系数 k
# ============================================================

def scale_scores(df: pd.DataFrame, k: float = DEFAULT_SCORE_K) -> pd.DataFrame:
    """
    将所有症状得分列乘以系数 k，放大其权重。

    k=2.0：原始得分 0~N → 0~2N，
    比 Z-score 特征幅值（~±3）略高，使模型更关注症状得分。
    """
    df = df.copy()
    score_cols = [c for c in df.columns if c.startswith(SCORE_PREFIX)]
    if not score_cols:
        return df

    logger.info(f"  [症状得分] {len(score_cols)} 列 × k={k}")
    for col in score_cols:
        df[col] = (df[col] * k).astype("int64")

    return df


# ============================================================
# 步骤 6：标签 one-hot 编码
# ============================================================

def _parse_disease_labels(value, known_diseases: List[str]) -> List[str]:
    """
    把单个样本的"患病情况"字符串解析成疾病列表。

    支持分隔符：分号、逗号、中文顿号。
    只保留 known_diseases 中的病名，其余过滤。
    """
    if value is None or pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s == "健康":
        return []

    for sep in [";", ",", "，"]:
        s = s.replace(sep, ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]

    allowed = set(known_diseases)
    return [p for p in parts if p in allowed]


def encode_one_hot_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    把"患病情况"列（"疾病A;疾病B;…"）拆成 19 个 0/1 列。

    设计：
        - 19 列顺序与 TARGET_DISEASES（diagnostic_rules）一致
        - 整数列（int8），不参与 Z-score
        - 未知病名 → 全部为 0（不视为患病）
    """
    df = df.copy()
    diseases = TARGET_DISEASES
    target_col = "患病情况"

    label_cols = [f"{LABEL_PREFIX}{d}" for d in diseases]
    for col in label_cols:
        df[col] = 0

    if target_col in df.columns:
        for i, val in df[target_col].items():
            hit = _parse_disease_labels(val, diseases)
            for d in hit:
                col = f"{LABEL_PREFIX}{d}"
                if col in label_cols:
                    df.at[i, col] = 1

    for col in label_cols:
        df[col] = df[col].astype("int8")

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df


# ============================================================
# 列重排
# ============================================================

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    列重排：
        所有标签列（LABEL_PREFIX 前缀）移到最后，其余列保持原顺序。
    """
    label_cols = [c for c in df.columns if c.startswith(LABEL_PREFIX)]
    other_cols = [c for c in df.columns if not c.startswith(LABEL_PREFIX)]
    return df[other_cols + label_cols] if label_cols else df


# ============================================================
# 主函数
# ============================================================

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    完整预处理管道。

    输入：stage01 处理后的 DataFrame（9374 × 40）
    输出：(预处理后的 DataFrame, 归一化统计量)
        DataFrame: 9374 × 78
            - 原始特征列 Z-score 标准化后
            + 19 个症状得分列（×k=2.0）
            + 19 个 one-hot 标签列
        norm_stats: 每列的 {"mean": float, "std": float}，供预测时复用
    """
    logger.info("[stage02] 开始预处理")
    logger.info(f"  输入 shape: {df.shape}")

    df = encode_bool_cols(df)       # 步骤 2
    df = encode_gender(df)          # 步骤 2
    df = compute_disease_scores(df)  # 步骤 3
    df, norm_stats = zscore_normalize(df)  # 步骤 4（返回统计量）
    df = scale_scores(df)            # 步骤 5
    df = encode_one_hot_target(df)   # 步骤 6
    df = reorder_columns(df)         # 列重排

    n_label = sum(1 for c in df.columns if c.startswith(LABEL_PREFIX))
    logger.info(f"  输出 shape: {df.shape}")
    logger.info(f"  特征列: {df.shape[1] - n_label}")
    logger.info(f"  标签列: {n_label} 个 one-hot 标签")
    logger.info(f"  归一化统计量: {len(norm_stats)} 列")

    return df, norm_stats


# ============================================================
# 调试入口
# ============================================================

def _summarize(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print("=" * 70)
    print("预处理前后对比")
    print("=" * 70)
    print(f"shape: {before.shape}  →  {after.shape}")
    print()

    score_cols = [c for c in after.columns if c.startswith(SCORE_PREFIX)]
    print(f"症状得分列 ({len(score_cols)} 个，乘 k={DEFAULT_SCORE_K} 后):")
    for col in score_cols[:5]:
        print(f"  {col}: min={after[col].min()}  max={after[col].max()}  "
              f"mean={after[col].mean():.2f}")
    if len(score_cols) > 5:
        print(f"  …（其余 {len(score_cols) - 5} 列省略）")

    label_cols = [c for c in after.columns if c.startswith(LABEL_PREFIX)]
    print()
    print(f"标签分布 (one-hot {len(label_cols)} 列):")
    for col in label_cols[:10]:
        n_pos = (after[col] == 1).sum()
        print(f"  {col}: 患病数={n_pos}  ({n_pos / len(after) * 100:.1f}%)")
    if len(label_cols) > 10:
        print(f"  …（其余 {len(label_cols) - 10} 列省略）")

    n_ill = after[label_cols].sum(axis=1)
    print(f"\n人均病种数: min={n_ill.min()}  max={n_ill.max()}  mean={n_ill.mean():.2f}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ml.stage01_load import load_and_prepare_data

    df_raw = load_and_prepare_data()
    df_clean, norm_stats = preprocess(df_raw)
    _summarize(df_raw, df_clean)

    _INTERMEDIATE_DIR = Path(__file__).resolve().parent.parent / "data" / "intermediate"
    _ROOT_DIR = Path(__file__).resolve().parent.parent / "data"
    _INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(_INTERMEDIATE_DIR / "stage02_cleaned.csv", index=False, encoding="utf-8-sig")
    logger.info(f"[stage02] 已落盘: {_INTERMEDIATE_DIR / 'stage02_cleaned.csv'}")
    df_clean.to_csv(_ROOT_DIR / "data_stage02.csv", index=False, encoding="utf-8-sig")
    logger.info(f"[stage02] 已落盘: {_ROOT_DIR / 'data_stage02.csv'}")

    # 落盘归一化统计量
    norm_path = _INTERMEDIATE_DIR / "stage02_norm_stats.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"[stage02] 归一化统计量已落盘: {norm_path}")
