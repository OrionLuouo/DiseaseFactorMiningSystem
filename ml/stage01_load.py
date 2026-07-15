"""
stage01_load.py — 数据加载与初步整理模块
========================================
职责：
    1. 从本地 CSV 文件读取病人数据（data/data_ill_final.csv）
    2. 删除无意义或高缺失的列（22 列）
    3. 按数据类型重排列：数值型在前，二值型在后，字符串型最后
    4. 将整理后的数据写入 data/data_stage01.csv

后续扩展点（先留好接口，不在本次实现）：
    - HDFS（需要 Hadoop + Spark）
    - Spark DataFrame 直读
    - 数据库（MySQL / PostgreSQL）

设计原则：
    1. 单一职责：只负责"读 + 初排"，不预处理、不训练。
    2. 默认参数全部走 config.yaml，调用方零参数就能用。
    3. 换数据集 / 换路径 → 改 config.yaml，不动代码。
    4. 每个函数都能被单独测试。

用法：
    from ml.stage01_load import load_and_prepare_data

    df = load_and_prepare_data()           # 走默认配置
    df = load_and_prepare_data(path="data/other.csv")   # 手动指定路径
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------ 路径常量 ------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "data_ill_final.csv"
_INTERMEDIATE_DIR = _PROJECT_ROOT / "data" / "intermediate"


def save_to_csv(
    df: pd.DataFrame,
    name: str,
    subdir: str = "intermediate",
    encoding: str = "utf-8-sig",
) -> Path:
    """【通用工具】把 DataFrame 保存到 data/<subdir>/<name>.csv。"""
    out_dir = _PROJECT_ROOT / "data" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    out_path = out_dir / name
    df.to_csv(out_path, index=False, encoding=encoding)
    logger.info(f"[stage01] 已落盘: {out_path} ({len(df)} 行 × {len(df.columns)} 列)")
    return out_path


# ============================================================
# 0. 列管理：删除 + 排序
# ============================================================

# 需删除的 22 列：ID、高缺失体检项、尿检重金属、生活方式专项
DROP_COLS: Set[str] = {
    # ID
    "SEQN",
    # 冠心病专项问卷（缺失率高且已被症状分覆盖）
    "是否在运动时胸痛",
    "疼痛能否在10分钟内缓解",
    # 体检项（缺失率高 / 与已有指标共线）
    "骨密度（g/cm2）",
    "体脂率（%）",
    "胰岛素（pmol/L）",
    # 生活习惯专项（缺失率过高）
    "平均每天中等以上强度锻炼时长（分钟）",
    "开始吸烟年龄",
    "家中是否有人吸烟",
    "一周内是否吸入过二手烟",
    # 饮食行为（无意义或已被覆盖）
    "是否会跳过正餐",
    # 家族史（已被疾病史覆盖）
    "父母是否患有骨质疏松",
    # 尿检重金属 11 项
    "尿检·钡（μg/L）",
    "尿检·镉（μg/L）",
    "尿检·钴（μg/L）",
    "尿检·铯（μg/L）",
    "尿检·锰（μg/L）",
    "尿检·钼（μg/L）",
    "尿检·铅（μg/L）",
    "尿检·锑（μg/L）",
    "尿检·锡（μg/L）",
    "尿检·铊（μg/L）",
    "尿检·钨（μg/L）",
}

# 二值型列（取值仅为 True / False / NaN），排在数值型之后
BOOL_COLS: Set[str] = {
    "脉搏是否规律",
    "是否有过胸部疼痛",
    "运动时是否气短",
    "存在家族糖尿病史",
    "饮食是否均衡",
    "是否从事体力劳动",
    "是否每周进行进行中等以上强度锻炼",
}


def _classify_columns(df: pd.DataFrame) -> List[str]:
    """
    将列按类型分组并返回排序后的列名列表。

    排序规则：
        1. 数值型（float64 / int64，且不在 BOOL_COLS 中）
        2. 二值型（在 BOOL_COLS 中，值为 True/False/NaN）
        3. 字符串型（object，如标签、性别）

    返回：
        排好序的列名列表。
    """
    numeric_cols: List[str] = []
    bool_cols: List[str] = []
    string_cols: List[str] = []

    for col in df.columns:
        if col == "患病情况":
            # 标签列始终放字符串区最前面
            string_cols.insert(0, col)
        elif col in BOOL_COLS:
            bool_cols.append(col)
        elif df[col].dtype in ("object", "str"):
            # object（字符串）或 pandas 扩展类型 str
            string_cols.append(col)
        else:
            # float64 / int64 → 数值型
            numeric_cols.append(col)

    return numeric_cols + bool_cols + string_cols


def _apply_drops(df: pd.DataFrame) -> pd.DataFrame:
    """删除 DROP_COLS 中存在的列，返回新 DataFrame。"""
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    if existing_drops:
        logger.info(f"[stage01] 删除 {len(existing_drops)} 列: {existing_drops}")
        df = df.drop(columns=existing_drops)
    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """按 数值型 → 二值型 → 字符串型 的顺序重排列。"""
    ordered = _classify_columns(df)
    return df[ordered]


# ============================================================
# 1. 配置加载（带兜底，不依赖 PyYAML）
# ============================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("[stage01] PyYAML 未安装，使用默认路径")
        return {}
    except Exception as e:
        logger.warning(f"[stage01] 读取 config.yaml 失败: {e}，使用默认路径")
        return {}


def _resolve_data_path(path: Optional[Union[str, Path]] = None) -> Path:
    """解析数据路径：显式传入 > config.yaml > 默认 data/data_ill_final.csv"""
    if path is not None:
        p = Path(path)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    cfg = _load_config()
    configured = cfg.get("data", {}).get("paths", {}).get("local")
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    return _DEFAULT_DATA_PATH


# ============================================================
# 2. 核心加载函数
# ============================================================

def load_patient_data(
    path: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    从本地 CSV 文件读取病人数据。

    参数：
        path:     CSV 文件路径。None 时按优先级自动解析。
        encoding: 文件编码，默认 utf-8。

    返回：
        pd.DataFrame，原始数据，未做任何清洗。

    异常：
        FileNotFoundError: 文件不存在
        ValueError:        文件为空或读取失败
    """
    resolved = _resolve_data_path(path)
    logger.info(f"[stage01] 读取病人数据: {resolved}")

    if not resolved.exists():
        raise FileNotFoundError(
            f"病人数据文件不存在: {resolved}\n"
            f"请检查路径，或在 config.yaml 的 data.paths.local 配置。"
        )

    try:
        df = pd.read_csv(resolved, encoding=encoding, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV 文件为空: {resolved}") from e

    if df.empty:
        raise ValueError(f"CSV 文件无有效数据: {resolved}")

    logger.info(
        f"[stage01] 读取完成: {len(df)} 行 × {len(df.columns)} 列"
    )
    return df


# ============================================================
# 3. 数据处理入口：删除 → 排序 → 落盘
# ============================================================

def load_and_prepare_data(
    path: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    完整 stage01 流程：读取 → 删除指定列 → 按类型重排列 → 保存。

    参数：
        path:        输入 CSV 路径，None 时走默认路径。
        encoding:    文件编码，默认 utf-8。
        output_path: 输出路径，None 时默认 data/data_stage01.csv。

    返回：
        处理后的 pd.DataFrame（数值型在前，二值型在后，字符串型最后）。
    """
    # ① 读取
    df = load_patient_data(path=path, encoding=encoding)

    # ② 删除列
    df = _apply_drops(df)

    # ③ 重排列：数值型 → 二值型 → 字符串型
    df = _reorder_columns(df)

    # ④ 落盘到 data/data_stage01.csv
    if output_path is None:
        output_path = _PROJECT_ROOT / "data" / "data_stage01.csv"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"[stage01] 已落盘: {output_path} "
        f"({len(df)} 行 × {len(df.columns)} 列)"
    )

    return df


# ============================================================
# 4. 调试入口
# ============================================================

def _summarize(df: pd.DataFrame) -> None:
    """打印数据概览，方便人工检查。"""
    print("=" * 70)
    print(f"数据形状: {df.shape}    (行 × 列)")
    print(f"输出文件: data/data_stage01.csv")
    print()
    print(f"数值型列 ({sum(1 for c in df.columns if df[c].dtype != 'object')}):")
    for col in df.columns:
        if df[col].dtype != "object":
            print(f"  {col}  ({df[col].dtype})")
    print()
    print(f"二值型列:")
    for col in df.columns:
        if col in BOOL_COLS:
            vc = df[col].value_counts(dropna=False).to_dict()
            print(f"  {col}: {vc}")
    print()
    print(f"字符串型列:")
    for col in df.columns:
        if df[col].dtype == "object" and col not in BOOL_COLS:
            vc = df[col].value_counts(dropna=False).head(3).to_dict()
            print(f"  {col}: {vc}")
    print()
    print("前 3 行:")
    print(df.head(3).to_string())
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_and_prepare_data()
    _summarize(df)
