"""
generate_synthetic_samples.py — 合成稀有病样本生成器
=====================================================

功能：
    为 9 个 F1=0 的稀有病生成医学上合理的合成样本，
    使模型能够学习到区分特征。

目标疾病及目标样本数：
    脂肪肝   200    肾衰竭   200
    骨质疏松 200    乙肝     200
    丙肝     200    痛风     200
    胆结石   200    肾结石   200
    肠胃炎   200
    共计 1800 条

设计原则：
    - 每个样本至少满足目标疾病的 2-3 个诊断规则条件
    - 保持与其他疾病的不一致性（避免过度共病）
    - 特征之间具有医学上合理的相关性（BMI 与体重/身高一致等）
    - 缺失值比例与原始数据相近（~30%）

使用：
    python ml/generate_synthetic_samples.py
    或
    from ml.generate_synthetic_samples import generate_all_disease_samples
    df_new = generate_all_disease_samples()
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 路径
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH = _PROJECT_ROOT / "data" / "data_ill_final.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "data_ill_final_augmented.csv"


# ============================================================
# 全局随机种子
# ============================================================
RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# 列名常量（与 data_ill_final.csv 完全一致）
# ============================================================
NUMERIC_COLS = [
    "饮酒频率",
    "血压收缩压（mmHg）",
    "血压舒张压（mmHg）",
    "脉搏（次/分钟）",
    "体重（kg）",
    "身高（cm）",
    "BMI（kg/m²）",
    "胆固醇（mg/dL）",
    "白细胞计数（1000个细胞/uL）",
    "红细胞计数（10⁶ 个细胞/µL）",
    "血红蛋白（g/dL）",
    "血小板计数（1000 个细胞/µL）",
    "每日食盐摄入程度",
    "每日热量摄入量（kcal）",
    "每日蛋白质摄入量（g）",
    "每日碳水摄入量（g）",
    "每日膳食纤维摄入量（g）",
    "每日脂肪摄入量（g）",
    "每日水摄入量（g）",
    "饮食是否充足",
    "起夜次数",
    "平均每天坐姿时长（分钟）",
    "工作日平均睡眠小时数",
    "打鼾频率",
    "葡萄糖（mg/dL）",
    "甘油三酯（mg/dL）",
    "肌酐（mg/dL）",
    "磷（mg/dL）",
    "钾（mmol/L）",
    "白蛋白（g/dL）",
    "血尿素氮（mg/dL）",
    "碳酸氢盐（mmol/L）",
    "钠（mmol/L）",
    "球蛋白（g/dL）",
]

BOOL_COLS = [
    "脉搏是否规律",
    "是否存在家族癌症史",
    "存在家族糖尿病史",
    "饮食是否均衡",
    "是否从事体力劳动",
    "是否每周进行进行中等以上强度锻炼",
]

ALL_COLS = (
    ["患病情况"]
    + NUMERIC_COLS
    + ["性别"]
    + BOOL_COLS
)


# ============================================================
# 基础分布参数（基于原始数据分析）
# ============================================================

def _get_base_params(gender: str) -> Dict[str, Tuple[float, float]]:
    """
    返回每个数值列的 (均值, 标准差)，
    来自原始数据的实际统计。
    """
    if gender == "男":
        return {
            "饮酒频率":              (2.0, 1.2),
            "血压收缩压（mmHg）":    (125, 18),
            "血压舒张压（mmHg）":    (78, 12),
            "脉搏（次/分钟）":        (74, 12),
            "体重（kg）":            (75, 15),
            "身高（cm）":            (170, 7),
            "BMI（kg/m²）":          (25.8, 4.5),
            "胆固醇（mg/dL）":        (185, 45),
            "白细胞计数（1000个细胞/uL）":  (6.8, 2.0),
            "红细胞计数（10⁶ 个细胞/µL）": (4.9, 0.5),
            "血红蛋白（g/dL）":       (14.5, 1.5),
            "血小板计数（1000 个细胞/µL）": (240, 60),
            "每日食盐摄入程度":       (2.5, 1.0),
            "每日热量摄入量（kcal）": (2100, 500),
            "每日蛋白质摄入量（g）":  (78, 25),
            "每日碳水摄入量（g）":    (260, 80),
            "每日膳食纤维摄入量（g）": (14, 6),
            "每日脂肪摄入量（g）":    (72, 25),
            "每日水摄入量（g）":      (1800, 800),
            "饮食是否充足":           (2.5, 0.8),
            "起夜次数":              (1.0, 1.2),
            "平均每天坐姿时长（分钟）": (360, 180),
            "工作日平均睡眠小时数":    (7.0, 1.5),
            "打鼾频率":              (1.5, 1.2),
            "葡萄糖（mg/dL）":        (100, 25),
            "甘油三酯（mg/dL）":      (120, 70),
            "肌酐（mg/dL）":         (0.9, 0.25),
            "磷（mg/dL）":           (3.5, 0.6),
            "钾（mmol/L）":          (4.3, 0.5),
            "白蛋白（g/dL）":         (4.2, 0.4),
            "血尿素氮（mg/dL）":      (14, 5),
            "碳酸氢盐（mmol/L）":     (24, 3),
            "钠（mmol/L）":          (141, 4),
            "球蛋白（g/dL）":         (2.9, 0.5),
        }
    else:
        return {
            "饮酒频率":              (1.5, 0.8),
            "血压收缩压（mmHg）":    (115, 18),
            "血压舒张压（mmHg）":    (73, 12),
            "脉搏（次/分钟）":        (78, 12),
            "体重（kg）":            (62, 13),
            "身高（cm）":            (158, 6),
            "BMI（kg/m²）":          (24.8, 4.8),
            "胆固醇（mg/dL）":        (190, 45),
            "白细胞计数（1000个细胞/uL）":  (6.5, 1.8),
            "红细胞计数（10⁶ 个细胞/µL）": (4.4, 0.5),
            "血红蛋白（g/dL）":       (13.0, 1.5),
            "血小板计数（1000 个细胞/µL）": (260, 65),
            "每日食盐摄入程度":       (2.2, 1.0),
            "每日热量摄入量（kcal）": (1750, 400),
            "每日蛋白质摄入量（g）":  (65, 20),
            "每日碳水摄入量（g）":    (220, 70),
            "每日膳食纤维摄入量（g）": (13, 5),
            "每日脂肪摄入量（g）":    (62, 22),
            "每日水摄入量（g）":      (1650, 700),
            "饮食是否充足":           (2.5, 0.8),
            "起夜次数":              (1.2, 1.3),
            "平均每天坐姿时长（分钟）": (350, 170),
            "工作日平均睡眠小时数":    (7.2, 1.5),
            "打鼾频率":              (1.0, 1.0),
            "葡萄糖（mg/dL）":        (95, 22),
            "甘油三酯（mg/dL）":      (105, 55),
            "肌酐（mg/dL）":         (0.75, 0.2),
            "磷（mg/dL）":           (3.6, 0.6),
            "钾（mmol/L）":          (4.2, 0.5),
            "白蛋白（g/dL）":         (4.0, 0.4),
            "血尿素氮（mg/dL）":      (13, 4),
            "碳酸氢盐（mmol/L）":     (24, 3),
            "钠（mmol/L）":          (141, 4),
            "球蛋白（g/dL）":         (3.0, 0.5),
        }


# ============================================================
# 各疾病的诊断规则（与 diagnostic_rules.py 一致）
# ============================================================

def _satisfies_脂肪肝(row) -> int:
    s = 0
    bmi = row.get("BMI（kg/m²）")
    tg = row.get("甘油三酯（mg/dL）")
    chol = row.get("胆固醇（mg/dL）")
    if bmi is not None and not np.isnan(bmi) and bmi >= 30: s += 1
    if tg is not None and not np.isnan(tg) and tg >= 200: s += 1
    if chol is not None and not np.isnan(chol) and chol >= 220: s += 1
    return s

def _satisfies_骨质疏松(row) -> int:
    s = 0
    bmi = row.get("BMI（kg/m²）")
    protein = row.get("每日蛋白质摄入量（g）")
    exercise = row.get("是否每周进行进行中等以上强度锻炼")
    if bmi is not None and not np.isnan(bmi) and bmi < 22: s += 1
    if protein is not None and not np.isnan(protein) and protein < 60: s += 1
    if exercise is not None and exercise in (False, 0): s += 1
    return s

def _satisfies_肾衰竭(row) -> int:
    s = 0
    cr = row.get("肌酐（mg/dL）")
    bun = row.get("血尿素氮（mg/dL）")
    noct = row.get("起夜次数")
    if cr is not None and not np.isnan(cr) and cr >= 1.5: s += 1
    if bun is not None and not np.isnan(bun) and bun >= 20: s += 1
    if noct is not None and not np.isnan(noct) and noct >= 2: s += 1
    return s

def _satisfies_乙肝(row) -> int:
    s = 0
    alb = row.get("白蛋白（g/dL）")
    glob = row.get("球蛋白（g/dL）")
    balanced = row.get("饮食是否均衡")
    if alb is not None and not np.isnan(alb) and alb < 4.0: s += 1
    if glob is not None and not np.isnan(glob) and glob >= 3.2: s += 1
    if balanced is not None and balanced in (False, 0): s += 1
    return s

def _satisfies_丙肝(row) -> int:
    s = 0
    alb = row.get("白蛋白（g/dL）")
    glob = row.get("球蛋白（g/dL）")
    balanced = row.get("饮食是否均衡")
    if alb is not None and not np.isnan(alb) and alb < 4.0: s += 1
    if glob is not None and not np.isnan(glob) and glob >= 3.2: s += 1
    if balanced is not None and balanced in (False, 0): s += 1
    return s

def _satisfies_痛风(row) -> int:
    s = 0
    bmi = row.get("BMI（kg/m²）")
    protein = row.get("每日蛋白质摄入量（g）")
    fat = row.get("每日脂肪摄入量（g）")
    balanced = row.get("饮食是否均衡")
    if bmi is not None and not np.isnan(bmi) and bmi >= 30: s += 1
    if protein is not None and not np.isnan(protein) and protein >= 100: s += 1
    if fat is not None and not np.isnan(fat) and fat >= 90: s += 1
    if balanced is not None and balanced in (False, 0): s += 1
    return s

def _satisfies_胆结石(row) -> int:
    s = 0
    bmi = row.get("BMI（kg/m²）")
    chol = row.get("胆固醇（mg/dL）")
    balanced = row.get("饮食是否均衡")
    if bmi is not None and not np.isnan(bmi) and bmi >= 30: s += 1
    if chol is not None and not np.isnan(chol) and chol >= 200: s += 1
    if balanced is not None and balanced in (False, 0): s += 1
    return s

def _satisfies_肾结石(row) -> int:
    s = 0
    noct = row.get("起夜次数")
    water = row.get("每日水摄入量（g）")
    bmi = row.get("BMI（kg/m²）")
    if noct is not None and not np.isnan(noct) and noct >= 2: s += 1
    if water is not None and not np.isnan(water) and water < 1500: s += 1
    if bmi is not None and not np.isnan(bmi) and bmi >= 30: s += 1
    return s

def _satisfies_肠胃炎(row) -> int:
    s = 0
    wbc = row.get("白细胞计数（1000个细胞/uL）")
    balanced = row.get("饮食是否均衡")
    if wbc is not None and not np.isnan(wbc) and wbc >= 8.0: s += 1
    if balanced is not None and balanced in (False, 0): s += 1
    return s


# ============================================================
# 辅助函数
# ============================================================

def _clip(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))


def _maybe_nan(val: float, prob: float = 0.25) -> Optional[float]:
    """以一定概率返回 NaN，保持原始值的自然感"""
    if random.random() < prob:
        return np.nan
    return val


def _maybe_nan_low(val: float) -> Optional[float]:
    """低缺失率（10%），用于关键诊断特征"""
    return _maybe_nan(val, prob=0.10)


def _maybe_nan_very_low(val: float) -> Optional[float]:
    """极低缺失率（5%），用于核心诊断特征"""
    return _maybe_nan(val, prob=0.05)


def _maybe_nan_high(val: float) -> Optional[float]:
    """较高缺失率（30%），用于饮食类特征"""
    return _maybe_nan(val, prob=0.30)


def _maybe_bool(val: bool, prob: float = 0.25) -> Optional[bool]:
    if random.random() < prob:
        return None
    return val


def _bool_to_optional(val: bool, nan_prob: float = 0.2) -> Optional[bool]:
    if random.random() < nan_prob:
        return None
    return val


# ============================================================
# 核心：生成单病样本
# ============================================================

def generate_脂肪肝_samples(n: int = 200) -> List[Dict]:
    """
    脂肪肝诊断规则：
        - BMI >= 30
        - 甘油三酯 >= 200
        - 胆固醇 >= 220
    生成策略：至少满足 2 个条件，保持其他特征正常
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "女"])
        base = _get_base_params(gender)

        # 满足规则的条件
        bmi = random.uniform(30, 38)
        tg = random.uniform(200, 400)
        chol = random.uniform(220, 320)

        # 推导一致的身高体重
        if gender == "男":
            height = random.uniform(165, 180)
        else:
            height = random.uniform(155, 168)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0] + 8, base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0] + 5, base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(76, 12)),
            "脉搏是否规律":          _maybe_nan_low(True),
            "体重（kg）":            _maybe_nan_very_low(weight),
            "身高（cm）":            _maybe_nan_very_low(height),
            "BMI（kg/m²）":          _maybe_nan_very_low(bmi),
            "胆固醇（mg/dL）":        _maybe_nan_very_low(chol),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.8, 1.8)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0], 1.2)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(240, 55)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan_low(random.random() < 0.25),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(2.8, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(2000, 3200)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(70, 110)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(250, 400)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(10, 22)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(80, 130)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1500, 2500)),
            "饮食是否充足":           _maybe_nan(random.gauss(2.5, 0.8)),
            "饮食是否均衡":          _maybe_nan(random.random() < 0.4),
            "起夜次数":              _maybe_nan(random.gauss(1.2, 1.2)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.35),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.35),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(200, 480)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(5.5, 9.0)),
            "打鼾频率":              _maybe_nan(random.uniform(1, 4)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(95, 145)),
            "甘油三酯（mg/dL）":      _maybe_nan_very_low(tg),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.9, 0.2)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.5, 0.5)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.3, 0.4)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(4.0, 0.4)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(14, 4)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(141, 4)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(2.9, 0.5)),
        }
        row["患病情况"] = "脂肪肝"
        rows.append(row)
    return rows


def generate_骨质疏松_samples(n: int = 200) -> List[Dict]:
    """
    骨质疏松诊断规则：
        - BMI < 22
        - 每日蛋白质摄入量 < 60g
        - 不进行中等以上强度锻炼
    """
    rows = []
    for i in range(n):
        gender = random.choice(["女", "女", "男"])  # 女性为主
        base = _get_base_params(gender)

        bmi = random.uniform(17, 22)
        protein = random.uniform(35, 60)
        if gender == "男":
            height = random.uniform(160, 175)
        else:
            height = random.uniform(150, 163)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0] - 5, base["血压舒张压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0] - 3, base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(74, 10)),
            "脉搏是否规律":          _maybe_nan(True),
            "体重（kg）":            _maybe_nan(weight),
            "身高（cm）":            _maybe_nan_very_low(height),
            "BMI（kg/m²）":          _maybe_nan_very_low(bmi),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(180, 40)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.2, 1.6)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.4)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0] - 1.0, 1.2)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(245, 60)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan_low(random.random() < 0.2),
            "每日食盐摄入程度":       _maybe_nan_high(random.gauss(2.0, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan_high(random.uniform(1200, 1800)),
            "每日蛋白质摄入量（g）":  _maybe_nan_very_low(protein),
            "每日碳水摄入量（g）":    _maybe_nan_high(random.uniform(150, 250)),
            "每日膳食纤维摄入量（g）": _maybe_nan_high(random.uniform(8, 16)),
            "每日脂肪摄入量（g）":    _maybe_nan_high(random.uniform(40, 75)),
            "每日水摄入量（g）":      _maybe_nan_high(random.uniform(1200, 2000)),
            "饮食是否充足":           _maybe_nan_high(random.uniform(1.5, 3.0)),
            "饮食是否均衡":          _maybe_nan_low(random.random() < 0.5),
            "起夜次数":              _maybe_nan_high(random.gauss(1.0, 1.0)),
            "是否从事体力劳动":       _maybe_nan_low(random.random() < 0.4),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan_low(False),  # 主要满足条件
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(300, 480)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(7.0, 9.0)),
            "打鼾频率":              _maybe_nan(random.uniform(0, 2)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(88, 115)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(80, 160)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.8, 0.2)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.3, 0.5)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.1, 0.4)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(3.8, 0.4)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(13, 4)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(140, 4)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(2.9, 0.5)),
        }
        row["患病情况"] = "骨质疏松"
        rows.append(row)
    return rows


def generate_肾衰竭_samples(n: int = 200) -> List[Dict]:
    """
    肾衰竭诊断规则：
        - 肌酐 >= 1.5 mg/dL
        - 血尿素氮 >= 20 mg/dL
        - 起夜次数 >= 2
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "女"])
        base = _get_base_params(gender)

        cr = random.uniform(1.5, 4.0)
        bun = random.uniform(20, 50)
        noct = random.randint(2, 5)

        if gender == "男":
            height = random.uniform(165, 178)
        else:
            height = random.uniform(153, 167)
        bmi = random.uniform(22, 32)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0] + 12, base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0] + 8, base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(78, 12)),
            "脉搏是否规律":          _maybe_nan(random.random() < 0.8),
            "体重（kg）":            _maybe_nan(weight),
            "身高（cm）":            _maybe_nan(height),
            "BMI（kg/m²）":          _maybe_nan(bmi),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(190, 45)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(7.2, 2.0)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0] - 0.3, 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0] - 1.5, 1.3)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(220, 60)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.3),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(3.0, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1600, 2400)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(55, 95)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(200, 320)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(10, 20)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(60, 100)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1200, 2200)),
            "饮食是否充足":           _maybe_nan(random.uniform(2.0, 3.0)),
            "饮食是否均衡":          _maybe_nan(random.random() < 0.4),
            "起夜次数":              _maybe_nan_very_low(float(noct)),
            "是否从事体力劳动":       _maybe_nan_low(random.random() < 0.3),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan_low(random.random() < 0.25),
            "平均每天坐姿时长（分钟）": _maybe_nan_high(random.uniform(250, 450)),
            "工作日平均睡眠小时数":    _maybe_nan_high(random.uniform(5.5, 8.5)),
            "打鼾频率":              _maybe_nan_high(random.uniform(1, 3)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(95, 150)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(100, 220)),
            "肌酐（mg/dL）":         _maybe_nan_very_low(cr),
            "磷（mg/dL）":           _maybe_nan(random.gauss(4.2, 1.0)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.8, 0.8)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(3.6, 0.5)),
            "血尿素氮（mg/dL）":      _maybe_nan_very_low(bun),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(22, 4)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(139, 5)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(3.1, 0.6)),
        }
        row["患病情况"] = "肾衰竭"
        rows.append(row)
    return rows


def generate_乙肝_samples(n: int = 200) -> List[Dict]:
    """
    乙肝诊断规则：
        - 白蛋白 < 4.0 g/dL
        - 球蛋白 >= 3.2 g/dL
        - 饮食不均衡
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "女"])
        base = _get_base_params(gender)

        alb = random.uniform(2.8, 4.0)
        glob = random.uniform(3.2, 4.5)

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0] + 0.5, base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0], base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0], base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(75, 11)),
            "脉搏是否规律":          _maybe_nan(True),
            "体重（kg）":            _maybe_nan(random.gauss(base["体重（kg）"][0], base["体重（kg）"][1])),
            "身高（cm）":            _maybe_nan(random.gauss(base["身高（cm）"][0], base["身高（cm）"][1])),
            "BMI（kg/m²）":          _maybe_nan(random.gauss(base["BMI（kg/m²）"][0], base["BMI（kg/m²）"][1])),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(175, 42)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.5, 1.8)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0] - 0.5, 1.3)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(200, 65)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.2),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(2.5, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1500, 2300)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(55, 90)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(180, 300)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(9, 18)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(55, 95)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1300, 2200)),
            "饮食是否充足":           _maybe_nan(random.uniform(2.0, 3.0)),
            "饮食是否均衡":          _maybe_nan_low(False),  # 主要满足条件
            "起夜次数":              _maybe_nan(random.gauss(1.2, 1.2)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.35),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.3),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(280, 450)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(6.0, 8.5)),
            "打鼾频率":              _maybe_nan(random.uniform(0, 3)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(88, 125)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(90, 180)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.9, 0.25)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.5, 0.5)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.2, 0.5)),
            "白蛋白（g/dL）":         _maybe_nan_very_low(alb),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(14, 5)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(141, 4)),
            "球蛋白（g/dL）":         _maybe_nan_very_low(glob),
        }
        row["患病情况"] = "乙肝"
        rows.append(row)
    return rows


def generate_丙肝_samples(n: int = 200) -> List[Dict]:
    """
    丙肝诊断规则：与乙肝相同
        - 白蛋白 < 4.0 g/dL
        - 球蛋白 >= 3.2 g/dL
        - 饮食不均衡
    与乙肝的差异：丙肝患者的代谢指标稍有不同
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "女"])
        base = _get_base_params(gender)

        alb = random.uniform(2.6, 4.0)
        glob = random.uniform(3.2, 5.0)

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0] + 0.8, base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0], base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0], base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(76, 12)),
            "脉搏是否规律":          _maybe_nan(random.random() < 0.8),
            "体重（kg）":            _maybe_nan(random.gauss(base["体重（kg）"][0] - 3, base["体重（kg）"][1])),
            "身高（cm）":            _maybe_nan(random.gauss(base["身高（cm）"][0], base["身高（cm）"][1])),
            "BMI（kg/m²）":          _maybe_nan(random.gauss(base["BMI（kg/m²）"][0] - 1.5, base["BMI（kg/m²）"][1])),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(170, 45)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.0, 2.0)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0] - 0.4, 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0] - 1.0, 1.4)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(180, 65)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.2),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(2.5, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1400, 2200)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(50, 85)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(170, 280)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(8, 17)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(50, 90)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1200, 2100)),
            "饮食是否充足":           _maybe_nan(random.uniform(2.0, 3.0)),
            "饮食是否均衡":          _maybe_nan(False, 0.05),
            "起夜次数":              _maybe_nan(random.gauss(1.3, 1.3)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.3),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.25),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(280, 450)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(6.0, 8.5)),
            "打鼾频率":              _maybe_nan(random.uniform(0, 3)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(90, 135)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(80, 200)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.85, 0.22)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.4, 0.6)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.1, 0.5)),
            "白蛋白（g/dL）":         _maybe_nan_very_low(alb),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(13, 5)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(140, 4)),
            "球蛋白（g/dL）":         _maybe_nan_very_low(glob),
        }
        row["患病情况"] = "丙肝"
        rows.append(row)
    return rows


def generate_痛风_samples(n: int = 200) -> List[Dict]:
    """
    痛风诊断规则：
        - BMI >= 30
        - 每日蛋白质摄入量 >= 100g
        - 每日脂肪摄入量 >= 90g
        - 饮食不均衡
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "男", "女"])
        base = _get_base_params(gender)

        bmi = random.uniform(30, 40)
        protein = random.uniform(100, 160)
        fat = random.uniform(90, 150)
        cal = protein * 4 + fat * 9 + 300  # 估算热量

        if gender == "男":
            height = random.uniform(165, 182)
        else:
            height = random.uniform(155, 170)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.uniform(2, 5)),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(130, 18)),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(82, 12)),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(78, 12)),
            "脉搏是否规律":          _maybe_nan(True),
            "体重（kg）":            _maybe_nan(weight),
            "身高（cm）":            _maybe_nan(height),
            "BMI（kg/m²）":          _maybe_nan(bmi),
            "胆固醇（mg/dL）":        _maybe_nan(random.uniform(190, 280)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(7.0, 2.0)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0], 1.5)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(245, 60)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.3),
            "每日食盐摄入程度":       _maybe_nan(random.uniform(3, 5)),
            "每日热量摄入量（kcal）": _maybe_nan(cal),
            "每日蛋白质摄入量（g）":  _maybe_nan(protein),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(200, 380)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(10, 20)),
            "每日脂肪摄入量（g）":    _maybe_nan(fat),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1200, 2000)),
            "饮食是否充足":           _maybe_nan(random.uniform(2.5, 3.5)),
            "饮食是否均衡":          _maybe_nan(False, 0.05),
            "起夜次数":              _maybe_nan(random.gauss(1.5, 1.2)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.4),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.2),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(250, 480)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(6.0, 8.0)),
            "打鼾频率":              _maybe_nan(random.uniform(2, 4)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(95, 140)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(120, 250)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(1.0, 0.3)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.6, 0.6)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.4, 0.5)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(4.1, 0.4)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.uniform(15, 28)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(142, 4)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(3.0, 0.5)),
        }
        row["患病情况"] = "痛风"
        rows.append(row)
    return rows


def generate_胆结石_samples(n: int = 200) -> List[Dict]:
    """
    胆结石诊断规则：
        - BMI >= 30
        - 胆固醇 >= 200
        - 饮食不均衡
    """
    rows = []
    for i in range(n):
        gender = random.choice(["女", "女", "男"])  # 女性发病率更高
        base = _get_base_params(gender)

        bmi = random.uniform(30, 38)
        chol = random.uniform(200, 320)

        if gender == "男":
            height = random.uniform(163, 178)
        else:
            height = random.uniform(153, 168)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0] + 5, base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0] + 3, base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(75, 11)),
            "脉搏是否规律":          _maybe_nan_low(True),
            "体重（kg）":            _maybe_nan_very_low(weight),
            "身高（cm）":            _maybe_nan_very_low(height),
            "BMI（kg/m²）":          _maybe_nan_very_low(bmi),
            "胆固醇（mg/dL）":        _maybe_nan_very_low(chol),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.8, 1.8)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0], 1.3)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(240, 60)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan_low(random.random() < 0.25),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(2.5, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1800, 2800)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(65, 100)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(220, 360)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(9, 18)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(70, 120)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1300, 2200)),
            "饮食是否充足":           _maybe_nan(random.uniform(2.5, 3.0)),
            "饮食是否均衡":          _maybe_nan(False, 0.05),
            "起夜次数":              _maybe_nan(random.gauss(1.2, 1.2)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.35),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.3),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(280, 450)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(6.5, 8.5)),
            "打鼾频率":              _maybe_nan(random.uniform(1, 3)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(90, 130)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(110, 220)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.88, 0.2)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.5, 0.5)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.2, 0.4)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(4.0, 0.4)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(14, 4)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(141, 4)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(3.0, 0.5)),
        }
        row["患病情况"] = "胆结石"
        rows.append(row)
    return rows


def generate_肾结石_samples(n: int = 200) -> List[Dict]:
    """
    肾结石诊断规则：
        - 起夜次数 >= 2
        - 每日水摄入量 < 1500g
        - BMI >= 30
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "男", "女"])
        base = _get_base_params(gender)

        noct = random.randint(2, 5)
        water = random.uniform(500, 1500)
        bmi = random.uniform(28, 38)

        if gender == "男":
            height = random.uniform(165, 180)
        else:
            height = random.uniform(154, 168)
        weight = bmi * (height / 100) ** 2

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0], base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0], base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.gauss(76, 12)),
            "脉搏是否规律":          _maybe_nan(True),
            "体重（kg）":            _maybe_nan(weight),
            "身高（cm）":            _maybe_nan(height),
            "BMI（kg/m²）":          _maybe_nan(bmi),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(188, 45)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(random.gauss(6.8, 1.8)),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.5)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0], 1.3)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(245, 60)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.2),
            "每日食盐摄入程度":       _maybe_nan(random.uniform(2, 5)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1700, 2600)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(70, 110)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(200, 340)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(10, 20)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(65, 110)),
            "每日水摄入量（g）":      _maybe_nan(water),
            "饮食是否充足":           _maybe_nan(random.uniform(2.0, 3.0)),
            "饮食是否均衡":          _maybe_nan(random.random() < 0.5),
            "起夜次数":              _maybe_nan(float(noct)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.35),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.3),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(280, 450)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(6.0, 8.5)),
            "打鼾频率":              _maybe_nan(random.uniform(1, 3)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(90, 125)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(100, 200)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(1.0, 0.3)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.6, 0.6)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.3, 0.5)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(4.1, 0.4)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.uniform(15, 28)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.gauss(24, 3)),
            "钠（mmol/L）":          _maybe_nan(random.gauss(142, 5)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(2.9, 0.5)),
        }
        row["患病情况"] = "肾结石"
        rows.append(row)
    return rows


def generate_肠胃炎_samples(n: int = 200) -> List[Dict]:
    """
    肠胃炎诊断规则：
        - 白细胞计数 >= 8.0（炎症）
        - 饮食不均衡
    """
    rows = []
    for i in range(n):
        gender = random.choice(["男", "女"])
        base = _get_base_params(gender)

        wbc = random.uniform(8.0, 15.0)

        row = {
            "饮酒频率":              _maybe_nan(random.gauss(base["饮酒频率"][0], base["饮酒频率"][1])),
            "血压收缩压（mmHg）":    _maybe_nan(random.gauss(base["血压收缩压（mmHg）"][0] - 5, base["血压收缩压（mmHg）"][1])),
            "血压舒张压（mmHg）":    _maybe_nan(random.gauss(base["血压舒张压（mmHg）"][0] - 5, base["血压舒张压（mmHg）"][1])),
            "脉搏（次/分钟）":        _maybe_nan(random.uniform(85, 110)),
            "脉搏是否规律":          _maybe_nan(True),
            "体重（kg）":            _maybe_nan(random.gauss(base["体重（kg）"][0], base["体重（kg）"][1])),
            "身高（cm）":            _maybe_nan(random.gauss(base["身高（cm）"][0], base["身高（cm）"][1])),
            "BMI（kg/m²）":          _maybe_nan(random.gauss(base["BMI（kg/m²）"][0], base["BMI（kg/m²）"][1])),
            "胆固醇（mg/dL）":        _maybe_nan(random.gauss(175, 42)),
            "白细胞计数（1000个细胞/uL）":  _maybe_nan(wbc),
            "红细胞计数（10⁶ 个细胞/µL）": _maybe_nan(random.gauss(base["红细胞计数（10⁶ 个细胞/µL）"][0], 0.4)),
            "血红蛋白（g/dL）":       _maybe_nan(random.gauss(base["血红蛋白（g/dL）"][0], 1.2)),
            "血小板计数（1000 个细胞/µL）": _maybe_nan(random.gauss(250, 65)),
            "性别": gender,
            "存在家族糖尿病史":       _maybe_nan(random.random() < 0.2),
            "每日食盐摄入程度":       _maybe_nan(random.gauss(2.2, 1.0)),
            "每日热量摄入量（kcal）": _maybe_nan(random.uniform(1200, 1900)),
            "每日蛋白质摄入量（g）":  _maybe_nan(random.uniform(45, 80)),
            "每日碳水摄入量（g）":    _maybe_nan(random.uniform(150, 260)),
            "每日膳食纤维摄入量（g）": _maybe_nan(random.uniform(8, 15)),
            "每日脂肪摄入量（g）":    _maybe_nan(random.uniform(40, 80)),
            "每日水摄入量（g）":      _maybe_nan(random.uniform(1000, 2000)),
            "饮食是否充足":           _maybe_nan(random.uniform(1.5, 2.5)),
            "饮食是否均衡":          _maybe_nan(False, 0.05),
            "起夜次数":              _maybe_nan(random.uniform(0, 4)),
            "是否从事体力劳动":       _maybe_nan(random.random() < 0.35),
            "是否每周进行进行中等以上强度锻炼": _maybe_nan(random.random() < 0.3),
            "平均每天坐姿时长（分钟）": _maybe_nan(random.uniform(280, 450)),
            "工作日平均睡眠小时数":    _maybe_nan(random.uniform(5.5, 8.0)),
            "打鼾频率":              _maybe_nan(random.uniform(0, 2)),
            "葡萄糖（mg/dL）":        _maybe_nan(random.uniform(85, 120)),
            "甘油三酯（mg/dL）":      _maybe_nan(random.uniform(80, 170)),
            "肌酐（mg/dL）":         _maybe_nan(random.gauss(0.85, 0.2)),
            "磷（mg/dL）":           _maybe_nan(random.gauss(3.5, 0.5)),
            "钾（mmol/L）":          _maybe_nan(random.gauss(4.0, 0.5)),
            "白蛋白（g/dL）":         _maybe_nan(random.gauss(3.8, 0.5)),
            "血尿素氮（mg/dL）":      _maybe_nan(random.gauss(13, 4)),
            "碳酸氢盐（mmol/L）":     _maybe_nan(random.uniform(20, 26)),
            "钠（mmol/L）":          _maybe_nan(random.uniform(135, 146)),
            "球蛋白（g/dL）":         _maybe_nan(random.gauss(3.0, 0.5)),
        }
        row["患病情况"] = "肠胃炎"
        rows.append(row)
    return rows


# ============================================================
# 汇总生成
# ============================================================

def generate_all_disease_samples(
    n_per_disease: int = 200,
) -> pd.DataFrame:
    """
    生成所有 9 个稀有病的合成样本。

    参数：
        n_per_disease: 每个病生成多少条（默认 200）

    返回：
        DataFrame，格式与 data_ill_final.csv 一致
    """
    logger.info(f"开始生成合成数据，每病 {n_per_disease} 条...")

    generators = {
        "脂肪肝":   generate_脂肪肝_samples,
        "骨质疏松": generate_骨质疏松_samples,
        "肾衰竭":   generate_肾衰竭_samples,
        "乙肝":     generate_乙肝_samples,
        "丙肝":     generate_丙肝_samples,
        "痛风":     generate_痛风_samples,
        "胆结石":   generate_胆结石_samples,
        "肾结石":   generate_肾结石_samples,
        "肠胃炎":   generate_肠胃炎_samples,
    }

    all_rows = []
    for disease, gen_fn in generators.items():
        rows = gen_fn(n_per_disease)
        all_rows.extend(rows)
        logger.info(f"  {disease}: 生成 {len(rows)} 条")

    df = pd.DataFrame(all_rows)

    # 调整列顺序与原始数据一致
    col_order = [
        "患病情况",
        "饮酒频率",
        "血压收缩压（mmHg）",
        "血压舒张压（mmHg）",
        "脉搏（次/分钟）",
        "脉搏是否规律",
        "体重（kg）",
        "身高（cm）",
        "BMI（kg/m²）",
        "胆固醇（mg/dL）",
        "白细胞计数（1000个细胞/uL）",
        "红细胞计数（10⁶ 个细胞/µL）",
        "血红蛋白（g/dL）",
        "血小板计数（1000 个细胞/µL）",
        "性别",
        "存在家族糖尿病史",
        "每日食盐摄入程度",
        "每日热量摄入量（kcal）",
        "每日蛋白质摄入量（g）",
        "每日碳水摄入量（g）",
        "每日膳食纤维摄入量（g）",
        "每日脂肪摄入量（g）",
        "每日水摄入量（g）",
        "饮食是否充足",
        "饮食是否均衡",
        "起夜次数",
        "是否从事体力劳动",
        "是否每周进行进行中等以上强度锻炼",
        "平均每天坐姿时长（分钟）",
        "工作日平均睡眠小时数",
        "打鼾频率",
        "葡萄糖（mg/dL）",
        "甘油三酯（mg/dL）",
        "肌酐（mg/dL）",
        "磷（mg/dL）",
        "钾（mmol/L）",
        "白蛋白（g/dL）",
        "血尿素氮（mg/dL）",
        "碳酸氢盐（mmol/L）",
        "钠（mmol/L）",
        "球蛋白（g/dL）",
    ]
    df = df[col_order]

    logger.info(f"合成数据生成完成: {len(df)} 行 × {len(df.columns)} 列")
    return df


def validate_synthetic_data(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    验证合成数据是否满足各疾病的诊断规则。
    """
    diseases_rules = {
        "脂肪肝":   _satisfies_脂肪肝,
        "骨质疏松": _satisfies_骨质疏松,
        "肾衰竭":   _satisfies_肾衰竭,
        "乙肝":     _satisfies_乙肝,
        "丙肝":     _satisfies_丙肝,
        "痛风":     _satisfies_痛风,
        "胆结石":   _satisfies_胆结石,
        "肾结石":   _satisfies_肾结石,
        "肠胃炎":   _satisfies_肠胃炎,
    }

    report = {}
    for disease, rule_fn in diseases_rules.items():
        sub = df[df["患病情况"] == disease]
        scores = sub.apply(rule_fn, axis=1)
        pass_at_least_2 = (scores >= 2).sum()
        pass_rate = pass_at_least_2 / len(sub) * 100 if len(sub) > 0 else 0
        report[disease] = {
            "total": len(sub),
            "pass_at_least_2": int(pass_at_least_2),
            "pass_rate": f"{pass_rate:.1f}%",
            "mean_score": f"{scores.mean():.2f}",
        }
    return report


def append_to_original(
    synthetic_df: pd.DataFrame,
    original_path: Path = _DATA_PATH,
    output_path: Path = _OUTPUT_PATH,
) -> None:
    """
    将合成数据追加到原始数据文件。

    流程：
        1. 读取原始 data_ill_final.csv
        2. 拼接合成数据
        3. 写入 data_ill_final_augmented.csv
        4. 备份原文件后替换
    """
    import shutil

    logger.info(f"读取原始数据: {original_path}")
    df_orig = pd.read_csv(original_path, encoding="utf-8-sig")
    logger.info(f"原始数据: {len(df_orig)} 行")

    # 拼接
    df_combined = pd.concat([df_orig, synthetic_df], ignore_index=True)
    logger.info(f"合并后: {len(df_combined)} 行")

    # 先写入临时文件
    temp_path = output_path
    df_combined.to_csv(temp_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存到: {temp_path}")

    # 备份原文件
    backup_path = original_path.with_suffix(".csv.backup")
    shutil.copy2(original_path, backup_path)
    logger.info(f"已备份原文件到: {backup_path}")

    # 替换原文件
    shutil.move(temp_path, original_path)
    logger.info(f"已替换原文件: {original_path}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. 生成合成数据
    df_synthetic = generate_all_disease_samples(n_per_disease=200)

    # 2. 验证
    report = validate_synthetic_data(df_synthetic)
    print("\n合成数据验证（满足 ≥2 条诊断规则的比例）：")
    print(f"{'疾病':<8} {'总数':>5} {'满足≥2':>7} {'通过率':>8} {'平均得分':>8}")
    print("-" * 42)
    for disease, stats in report.items():
        print(
            f"  {disease:<6} {stats['total']:>5} {stats['pass_at_least_2']:>7} "
            f"{stats['pass_rate']:>8} {stats['mean_score']:>8}"
        )

    # 3. 追加到原始数据
    print("\n是否将合成数据追加到原始 data_ill_final.csv？")
    print("  - 将创建备份: data_ill_final.csv.backup")
    print("  - 将生成: data_ill_final_augmented.csv（用于追加）")
    append_to_original(df_synthetic)
    print("\n完成！建议重新运行流水线：")
    print("  python ml/stage01_load.py  # 重新加载")
    print("  python ml/stage02_preprocess.py  # 重新预处理")
    print("  python ml/stage03_ML_train.py  # 重新训练")
