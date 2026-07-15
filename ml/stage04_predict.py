"""
stage04_predict.py — 新病人预测接口
=====================================
流程：
    1. 前端字典 → 原始特征 DataFrame
    2. 计算 19 个症状得分
    3. Z-score 标准化（使用训练时的 mean/std）
    4. 症状得分 × k=2.0 放大
    5. 模型预测 → 19 种病概率
    6. 病史校验后处理（可选）
    7. 返回 Top-3 结果 + 正常人对比数据

API:
    predict(user_dict, history=None) -> dict
    predict_batch(user_dicts, history_list=None) -> list[dict]

配置：
    字段映射   : FRONTEND_TO_MODEL
    病名映射   : HISTORY_FIELD_MAP（前端键 → 模型病名）
    正常人参考 : NORMAL_REFERENCE（用于前端雷达图 / 对比表）

依赖：
    - ml.diagnostic_rules（症状得分计算）
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MODEL_DIR = _PROJECT_ROOT / "ml" / "models"
_NORM_STATS_PATH = _PROJECT_ROOT / "data" / "intermediate" / "stage02_norm_stats.json"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "xgboost_chain_v1.joblib"

# ============================================================
# 病名常量（与 stage02 保持一致）
# ============================================================

TARGET_DISEASES: List[str] = [
    "高血压", "低血压", "睡眠呼吸暂停", "心律失常", "高尿酸血症",
    "糖尿病", "脂肪肝", "NAFLD", "痛风", "骨质疏松",
    "贫血", "感冒", "肠胃炎", "胆结石", "肾结石",
    "肾衰竭", "乙肝", "丙肝", "蛀牙",
]
SCORE_PREFIX: str = "症状得分_"
LABEL_PREFIX: str = "患病情况_"
DEFAULT_SCORE_K: float = 2.0

# ============================================================
# 前端字段 → 模型特征 映射表
# ============================================================
# key: 前端 JSON 中的字段名
# value: stage01 输出的列名（= stage02 标准化后的特征列名）

FRONTEND_TO_MODEL: Dict[str, str] = {
    # 基本信息
    "gender": "性别",
    # 生命体征
    "sbp": "血压收缩压（mmHg）",
    "dbp": "血压舒张压（mmHg）",
    "pulse": "脉搏（次/分钟）",
    # 体格指标
    "weight": "体重（kg）",
    "height": "身高（cm）",
    "bmi": "BMI（kg/m²）",
    # 血液检查
    "cholesterol": "胆固醇（mg/dL）",
    "white_blood_cells": "白细胞计数（1000个细胞/uL）",
    "red_blood_cells": "红细胞计数（10⁶ 个细胞/µL）",
    "hemoglobin": "血红蛋白（g/dL）",
    "platelets": "血小板计数（1000 个细胞/µL）",
    # 尿液 / 生化
    "glucose": "葡萄糖（mg/dL）",
    "triglycerides": "甘油三酯（mg/dL）",
    "creatinine": "肌酐（mg/dL）",
    "phosphorus": "磷（mg/dL）",
    "potassium": "钾（mmol/L）",
    "albumin": "白蛋白（g/dL）",
    "blood_urea_nitrogen": "血尿素氮（mg/dL）",
    "bicarbonate": "碳酸氢盐（mmol/L）",
    "sodium": "钠（mmol/L）",
    "globulin": "球蛋白（g/dL）",
    # 生活习惯
    "salt_intake": "每日食盐摄入程度",
    "daily_calories": "每日热量摄入量（kcal）",
    "daily_protein": "每日蛋白质摄入量（g）",
    "daily_carbs": "每日碳水摄入量（g）",
    "daily_fiber": "每日膳食纤维摄入量（g）",
    "daily_fat": "每日脂肪摄入量（g）",
    "daily_water": "每日水摄入量（g）",
    "alcohol_freq": "饮酒频率",
    "night_urination": "起夜次数",
    "sitting_hours": "平均每天坐姿时长（分钟）",
    "sleep_hours": "工作日平均睡眠小时数",
    "snoring_freq": "打鼾频率",
    # 布尔型
    "diet_sufficient": "饮食是否充足",
    "family_diabetes": "存在家族糖尿病史",
    "diet_balanced": "饮食是否均衡",
    "physical_labor": "是否从事体力劳动",
    "exercise": "是否每周进行进行中等以上强度锻炼",
    "pulse_regular": "脉搏是否规律",
}

# 布尔字段的白名单
BOOL_FIELDS: set = {
    "diet_sufficient", "family_diabetes", "diet_balanced",
    "physical_labor", "exercise", "pulse_regular",
}

# 性别映射
GENDER_MAP: Dict[str, int] = {"男": 1, "女": 0, "male": 1, "female": 0, "M": 1, "F": 0}

# ============================================================
# 病史字段映射（前端键 → 模型病名）
# ============================================================

HISTORY_FIELD_MAP: Dict[str, str] = {
    "hypertension": "高血压",
    "hypotension": "低血压",
    "sleep_apnea": "睡眠呼吸暂停",
    "arrhythmia": "心律失常",
    "hyperuricemia": "高尿酸血症",
    "diabetes": "糖尿病",
    "fatty_liver": "脂肪肝",
    "nafld": "NAFLD",
    "gout": "痛风",
    "osteoporosis": "骨质疏松",
    "anemia": "贫血",
    "common_cold": "感冒",
    "gastroenteritis": "肠胃炎",
    "gallstones": "胆结石",
    "kidney_stones": "肾结石",
    "kidney_failure": "肾衰竭",
    "hepatitis_b": "乙肝",
    "hepatitis_c": "丙肝",
    "cavities": "蛀牙",
}

# ============================================================
# 正常人参考值（用于前端雷达图 / 对比表）
# ============================================================

NORMAL_REFERENCE: Dict[str, float] = {
    "性别": 1,
    "血压收缩压（mmHg）": 120.0,
    "血压舒张压（mmHg）": 80.0,
    "脉搏（次/分钟）": 72.0,
    "体重（kg）": 65.0,
    "身高（cm）": 170.0,
    "BMI（kg/m²）": 22.0,
    "胆固醇（mg/dL）": 180.0,
    "白细胞计数（1000个细胞/uL）": 7.0,
    "红细胞计数（10⁶ 个细胞/µL）": 5.0,
    "血红蛋白（g/dL）": 14.0,
    "血小板计数（1000 个细胞/µL）": 250.0,
    "葡萄糖（mg/dL）": 90.0,
    "甘油三酯（mg/dL）": 120.0,
    "肌酐（mg/dL）": 1.0,
    "磷（mg/dL）": 3.5,
    "钾（mmol/L）": 4.0,
    "白蛋白（g/dL）": 4.5,
    "血尿素氮（mg/dL）": 15.0,
    "碳酸氢盐（mmol/L）": 24.0,
    "钠（mmol/L）": 140.0,
    "球蛋白（g/dL）": 2.5,
    "每日食盐摄入程度": 2,
    "每日热量摄入量（kcal）": 2000.0,
    "每日蛋白质摄入量（g）": 60.0,
    "每日碳水摄入量（g）": 250.0,
    "每日膳食纤维摄入量（g）": 25.0,
    "每日脂肪摄入量（g）": 65.0,
    "每日水摄入量（g）": 2000.0,
    "饮酒频率": 1,
    "起夜次数": 0,
    "平均每天坐姿时长（分钟）": 240.0,
    "工作日平均睡眠小时数": 7.5,
    "打鼾频率": 1,
    "饮食是否充足": 1,
    "存在家族糖尿病史": 0,
    "饮食是否均衡": 1,
    "是否从事体力劳动": 0,
    "是否每周进行进行中等以上强度锻炼": 1,
    "脉搏是否规律": 1,
}

# ============================================================
# 病史校验参数
# ============================================================

HISTORY_BOOST_MAX: float = 0.30      # 病史最多提升概率的 30%
HISTORY_FITNESS_HIGH: float = 1.0    # 模型概率 >= 阈值 → 高度吻合
HISTORY_FITNESS_MEDIUM: float = 0.5  # 模型概率 >= 阈值/2 → 中等吻合

# ============================================================
# 模型加载（全局缓存）
# ============================================================

_cached_model: Optional[Dict] = None


def _ensure_class_loaded() -> None:
    """
    确保 XGBoostClassifierChain 在 __main__ 中可用。
    解决：模型用 `python ml/stage03_ML_train.py` 训练时，pickle
    记录的模块名是 __main__；从 stage04 加载时必须把类注册到 __main__。
    """
    if "XGBoostClassifierChain" in getattr(sys.modules["__main__"], "__dict__", {}):
        return
    from ml.stage03_ML_train import XGBoostClassifierChain
    setattr(sys.modules["__main__"], "XGBoostClassifierChain", XGBoostClassifierChain)


def load_model(model_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    加载模型包（模型 + 阈值 + 病名列表 + 特征名 + 归一化统计量）。
    使用全局缓存，多次调用只加载一次。
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}。请先运行 ml/stage03_ML_train.py 训练模型。"
        )

    import joblib
    _ensure_class_loaded()
    logger.info(f"[stage04] 加载模型: {model_path}")
    obj = joblib.load(model_path)

    required_keys = {"model", "thresholds", "label_names", "feature_names"}
    missing = required_keys - obj.keys()
    if missing:
        raise ValueError(f"模型包缺少必要字段: {missing}")

    _cached_model = obj
    return _cached_model


# ============================================================
# 归一化统计量加载
# ============================================================

def load_norm_stats(path: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, float]]:
    """加载 stage02 输出的归一化统计量。"""
    if path is None:
        path = _NORM_STATS_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"归一化统计量文件不存在: {path}。请先运行 ml/stage02_preprocess.py。"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 前端字典 → 模型特征
# ============================================================

def _coerce_value(key: str, value: Any) -> Any:
    """
    类型转换：
        - 布尔字段: True/False/"true"/"false"/1/0 → 1/0
        - 性别字段: "男"/"女" → 1/0
        - 其余: 保持原值（期望数值型）
    """
    if key in BOOL_FIELDS:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            return 1 if value.lower() in ("true", "1", "是", "yes") else 0
        return int(bool(value))

    if key == "gender":
        if isinstance(value, str):
            return GENDER_MAP.get(value, value)
        return value

    return value


def frontend_to_model_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    将前端 JSON 字典转换为模型特征字典。

    步骤：
        1. 按 FRONTEND_TO_MODEL 映射字段名
        2. 类型转换（布尔、性别等）
        3. 缺失字段填 None

    返回: {模型列名: 转换后的值}
    """
    result: Dict[str, Any] = {}

    for frontend_key, model_col in FRONTEND_TO_MODEL.items():
        if frontend_key not in raw:
            result[model_col] = None
            continue
        result[model_col] = _coerce_value(frontend_key, raw[frontend_key])

    known_keys = set(FRONTEND_TO_MODEL.keys()) | {"history"}
    extra_keys = set(raw.keys()) - known_keys
    if extra_keys:
        logger.debug(f"[stage04] 忽略未识别的字段: {extra_keys}")

    return result


# ============================================================
# DataFrame 预处理（与 stage02 完全对齐）
# ============================================================

def _compute_disease_scores(df: pd.DataFrame) -> pd.DataFrame:
    """对单行/多行 DataFrame 计算症状得分。"""
    from ml.diagnostic_rules import get_diagnostic_rules_vec

    rules = get_diagnostic_rules_vec()
    for disease, score_fn in rules.items():
        col = f"{SCORE_PREFIX}{disease}"
        df[col] = score_fn(df).astype("int64")
    return df


def _apply_zscore(df: pd.DataFrame, norm_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    对数值特征列应用训练时的 Z-score 标准化。
    排除症状得分列、标签列。
    """
    df = df.copy()

    exclude = set()
    exclude |= {c for c in df.columns if c.startswith(SCORE_PREFIX)}
    exclude |= {c for c in df.columns if c.startswith(LABEL_PREFIX)}

    for col in df.columns:
        if col in exclude:
            continue
        if col not in norm_stats:
            logger.warning(f"[stage04] 特征 '{col}' 不在归一化统计量中，跳过标准化")
            continue
        stats = norm_stats[col]
        mean_val = stats["mean"]
        std_val = stats.get("std", 1.0)
        if std_val == 0:
            std_val = 1.0
        df[col] = (df[col] - mean_val) / std_val

    return df


def _scale_scores(df: pd.DataFrame, k: float = DEFAULT_SCORE_K) -> pd.DataFrame:
    """症状得分 × k。"""
    df = df.copy()
    score_cols = [c for c in df.columns if c.startswith(SCORE_PREFIX)]
    for col in score_cols:
        df[col] = (df[col] * k).astype("int64")
    return df


def build_feature_dataframe(
    raw: Dict[str, Any],
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    完整预处理流程：前端字典 → 模型输入 DataFrame。

    参数:
        raw: 前端原始字典
        norm_stats: 归一化统计量（None 时自动加载）

    返回: 单行 DataFrame，列顺序与训练时一致
    """
    if norm_stats is None:
        norm_stats = load_norm_stats()

    # 1. 字段映射 + 类型转换
    features = frontend_to_model_features(raw)

    # 2. 构建 DataFrame（1 行）
    df = pd.DataFrame([features])

    # 3. 计算症状得分
    df = _compute_disease_scores(df)

    # 4. Z-score 标准化
    df = _apply_zscore(df, norm_stats)

    # 5. 症状得分放大
    df = _scale_scores(df)

    # 6. 按模型保存的特征列顺序排列
    model_obj = load_model()
    feature_names = model_obj["feature_names"]

    # 填充缺失列为 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df


# ============================================================
# 病史解析
# ============================================================

def parse_history(history: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """
    解析前端病史字典。

    支持两种格式：
        {"高血压": 1, "糖尿病": 0, ...}  （中文病名，直接匹配）
        {"hypertension": 1, "diabetes": 0, ...}  （英文 key，需映射）

    返回: {模型病名: 0/1}
    """
    if not history:
        return {}

    result: Dict[str, int] = {}

    for key, value in history.items():
        if key in TARGET_DISEASES:
            result[key] = int(bool(value))
            continue

        disease_cn = HISTORY_FIELD_MAP.get(key)
        if disease_cn:
            result[disease_cn] = int(bool(value))
            continue

        logger.debug(f"[stage04] 忽略未知病史字段: {key}")

    return result


# ============================================================
# 病史校验后处理
# ============================================================

def _calc_fitness(prob: float, threshold: float) -> tuple[str, float]:
    """
    计算模型概率与病史的契合度。

    返回: (等级, 调整系数)
        等级: "high" | "medium" | "low"
        系数: 1.0 | 0.5 | 0.0
    """
    if prob >= threshold:
        return "high", HISTORY_FITNESS_HIGH
    elif prob >= threshold / 2:
        return "medium", HISTORY_FITNESS_MEDIUM
    else:
        return "low", 0.0


def apply_history_adjustment(
    probs: np.ndarray,
    history: Dict[str, int],
    thresholds: np.ndarray,
    label_names: List[str],
) -> tuple[np.ndarray, List[str]]:
    """
    病史校验后处理。

    原则：病史只能提升概率，不能降低；与症状矛盾的病史被忽略并警告。

    返回:
        (调整后概率, 警告列表)
    """
    adjusted = probs.copy()
    warnings: List[str] = []

    for i, disease in enumerate(label_names):
        user_has = history.get(disease, 0)
        if user_has == 0:
            continue

        fitness, factor = _calc_fitness(probs[i], thresholds[i])

        if fitness == "low":
            warnings.append(
                f"「{disease}」：症状指标不支持您声称的病史（模型概率 {probs[i]:.1%}），"
                f"建议到院复查。该病史未纳入本次预测。"
            )
            continue

        boost = (1.0 - probs[i]) * factor * HISTORY_BOOST_MAX
        adjusted[i] = min(probs[i] + boost, 1.0)

    return adjusted, warnings


# ============================================================
# 主预测函数
# ============================================================

def predict(
    user_dict: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    model_path: Optional[Union[str, Path]] = None,
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    对单个用户做疾病预测。

    参数:
        user_dict: 前端传入的原始字典（见 FRONTEND_TO_MODEL 键名）
        history: 病史字典（可选），格式：
                 - {"高血压": 1, "糖尿病": 0}  (中文)
                 - {"hypertension": 1, "diabetes": 0}  (英文 key)
        model_path: 模型文件路径（None 用默认路径）
        norm_stats: 归一化统计量（None 自动加载）
        top_n: 返回 Top-N 结果（默认 3）

    返回:
        {
            "top_disease": str,
            "confidence": float,
            "top_n": [{"rank": int, "disease": str, "probability": float}, ...],
            "warnings": [str, ...],
            "normal_reference": {字段名: 数值, ...},
        }
    """
    if top_n < 1:
        raise ValueError(f"top_n 必须 >= 1，当前: {top_n}")
    if top_n > len(TARGET_DISEASES):
        top_n = len(TARGET_DISEASES)

    # 1. 加载资源
    model_obj = load_model(model_path)
    model = model_obj["model"]
    thresholds = model_obj["thresholds"]
    label_names = model_obj["label_names"]

    if norm_stats is None:
        norm_stats = load_norm_stats()

    # 2. 构建特征
    df_features = build_feature_dataframe(user_dict, norm_stats=norm_stats)

    # 3. 模型预测
    proba = model.predict_proba(df_features.values)[0]

    # 4. 病史校验（可选）
    parsed_history = parse_history(history)
    warnings: List[str] = []
    if parsed_history:
        proba, warnings = apply_history_adjustment(
            proba, parsed_history, thresholds, label_names
        )

    # 5. 排序 + 截取 Top-N
    ranking = sorted(
        zip(label_names, proba),
        key=lambda x: x[1],
        reverse=True,
    )
    top_n_results = [
        {
            "rank": idx + 1,
            "disease": disease,
            "probability": round(float(prob), 4),
        }
        for idx, (disease, prob) in enumerate(ranking[:top_n])
    ]

    return {
        "top_disease": ranking[0][0],
        "confidence": round(float(ranking[0][1]), 4),
        "top_n": top_n_results,
        "warnings": warnings,
        "normal_reference": NORMAL_REFERENCE,
    }


def predict_batch(
    user_dicts: List[Dict[str, Any]],
    history_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    model_path: Optional[Union[str, Path]] = None,
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    批量预测。

    参数:
        user_dicts: 用户字典列表
        history_list: 对应的病史列表（长度需与 user_dicts 一致，None 表示无病史）
        model_path: 模型路径
        norm_stats: 归一化统计量
        top_n: Top-N

    返回: 每个用户的预测结果列表
    """
    if history_list is None:
        history_list = [None] * len(user_dicts)
    if len(history_list) != len(user_dicts):
        raise ValueError("user_dicts 和 history_list 长度不一致")

    return [
        predict(u, h, model_path=model_path, norm_stats=norm_stats, top_n=top_n)
        for u, h in zip(user_dicts, history_list)
    ]


# ============================================================
# 便捷：获取输入字段模板（供前端参考）
# ============================================================

def get_input_schema() -> Dict[str, Any]:
    """
    返回前端输入字段的说明，供前端开发参考。

    返回:
        {
            "fields": {前端字段: {"type": ..., "model_field": ...}},
            "history_fields": {前端病史字段: {"model_field": ...}},
        }
    """
    fields: Dict[str, Any] = {}
    for fe_key, model_col in FRONTEND_TO_MODEL.items():
        field_info: Dict[str, Any] = {"model_field": model_col}

        if fe_key == "gender":
            field_info["type"] = "string"
            field_info["options"] = ["男", "女"]
        elif fe_key in BOOL_FIELDS:
            field_info["type"] = "boolean"
        else:
            field_info["type"] = "number"

        fields[fe_key] = field_info

    history_fields = {}
    for fe_key, cn_name in HISTORY_FIELD_MAP.items():
        history_fields[fe_key] = {"model_field": cn_name}

    return {"fields": fields, "history_fields": history_fields}


# ============================================================
# 调试 / 测试入口
# ============================================================

if __name__ == "__main__":
    sys.path.insert(0, str(_SCRIPT_DIR.parent))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 示例输入
    sample_user = {
        "gender": "男",
        "sbp": 148,
        "dbp": 95,
        "pulse": 78,
        "weight": 82,
        "height": 172,
        "bmi": 27.7,
        "cholesterol": 240,
        "glucose": 105,
        "triglycerides": 180,
        "exercise": True,
        "snoring_freq": 3,
        "salt_intake": 3,
        "sleep_hours": 5.5,
    }

    sample_history = {
        "hypertension": 1,
    }

    print("=" * 60)
    print("stage04 预测测试")
    print("=" * 60)

    result = predict(sample_user, history=sample_history, top_n=3)
    print(f"\n最可能疾病: {result['top_disease']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"\nTop-{len(result['top_n'])} 结果:")
    for item in result["top_n"]:
        marker = " (高置信)" if item["probability"] >= 0.5 else ""
        print(f"  {item['rank']}. {item['disease']:<20} {item['probability']:.4f}{marker}")

    if result["warnings"]:
        print(f"\n警告:")
        for w in result["warnings"]:
            print(f"  ! {w}")

    print(f"\n正常人参考值 (key count={len(result['normal_reference'])})")
    print("  已包含，供前端绘制雷达图使用")
    print("=" * 60)

    # 测试恶意输入场景
    print("\n--- 病史冲突场景测试 ---")
    malicious = sample_user.copy()
    malicious["sbp"] = 110
    malicious["dbp"] = 70
    result2 = predict(malicious, history={"hypertension": 1}, top_n=3)
    print(f"血压正常但声称高血压 → 最可能疾病: {result2['top_disease']}")
    print(f"置信度: {result2['confidence']:.2%}")
    if result2["warnings"]:
        for w in result2["warnings"]:
            print(f"  ! {w}")
    print("=" * 60)
