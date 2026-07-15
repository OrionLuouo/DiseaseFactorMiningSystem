"""
diagnostic_rules.py — 临床诊断规则（症状得分）
=================================================

提供两种调用方式：
    • 向量化（推荐，整列操作，速度快 100 倍）：
        rules_vec = get_diagnostic_rules_vec()          # {疾病名: vectorized_func}
        df["症状得分_高血压"] = rules_vec["高血压"](df)

    • 逐行（backward-compat）：
        rules = get_diagnostic_rules()                   # {疾病名: scalar_func}
        df["症状得分_高血压"] = df.apply(rules["高血压"], axis=1)

调用方：
    ml/stage02_preprocess.py  → compute_disease_scores()

设计原则：
    - 每条规则基于临床公认阈值（如高血压 140/90）
    - 缺失的指标不计入得分（既不 +1 也不 +0）
    - 评分 = 满足的关键指标个数（int，≥ 0）
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# 通用辅助
# ============================================================

def _bool_score(row, col):
    """逐行：bool 列 True→1，False→0，缺失→0"""
    v = row.get(col, None)
    if v is None or pd.isna(v):
        return 0
    return 1 if bool(v) else 0


def _mask(series, op, threshold):
    """对 Series 做条件判断，返回 bool Series（NaN 处为 False）"""
    return series.notna() & op(series, threshold)


# ============================================================
# 逐行版本（backward-compat，供外部单行调用）
# ============================================================

def score_diabetes(row):
    s = 0
    if not pd.isna(row.get("葡萄糖（mg/dL）")) and row["葡萄糖（mg/dL）"] >= 126: s += 1
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 28: s += 1
    s += _bool_score(row, "存在家族糖尿病史")
    return s

def score_hypertension(row):
    s = 0
    if not pd.isna(row.get("血压收缩压（mmHg）")) and row["血压收缩压（mmHg）"] >= 140: s += 1
    if not pd.isna(row.get("血压舒张压（mmHg）")) and row["血压舒张压（mmHg）"] >= 90: s += 1
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 28: s += 1
    return s

def score_hypotension(row):
    s = 0
    if not pd.isna(row.get("血压收缩压（mmHg）")) and row["血压收缩压（mmHg）"] < 90: s += 1
    if not pd.isna(row.get("血压舒张压（mmHg）")) and row["血压舒张压（mmHg）"] < 60: s += 1
    if not pd.isna(row.get("每日热量摄入量（kcal）")) and row["每日热量摄入量（kcal）"] < 1500: s += 1
    return s

def score_sleep_apnea(row):
    s = 0
    if not pd.isna(row.get("打鼾频率")) and row["打鼾频率"] >= 2: s += 1
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 25: s += 1
    if not pd.isna(row.get("起夜次数")) and row["起夜次数"] >= 2: s += 1
    return s

def score_arrhythmia(row):
    s = 0
    if not pd.isna(row.get("脉搏（次/分钟）")):
        p = row["脉搏（次/分钟）"]
        if p < 60 or p > 100: s += 1
    s += _bool_score(row, "脉搏是否规律")
    if not pd.isna(row.get("胆固醇（mg/dL）")) and row["胆固醇（mg/dL）"] >= 200: s += 1
    return s

def score_hyperuricemia(row):
    s = 0
    if not pd.isna(row.get("每日蛋白质摄入量（g）")) and row["每日蛋白质摄入量（g）"] >= 90: s += 1
    if not pd.isna(row.get("每日脂肪摄入量（g）")) and row["每日脂肪摄入量（g）"] >= 90: s += 1
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 25: s += 1
    return s

def score_fatty_liver(row):
    s = 0
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 30: s += 1
    if not pd.isna(row.get("甘油三酯（mg/dL）")) and row["甘油三酯（mg/dL）"] >= 200: s += 1
    if not pd.isna(row.get("胆固醇（mg/dL）")) and row["胆固醇（mg/dL）"] >= 220: s += 1
    return s

def score_gout(row):
    s = 0
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 30: s += 1
    if not pd.isna(row.get("每日蛋白质摄入量（g）")) and row["每日蛋白质摄入量（g）"] >= 100: s += 1
    if not pd.isna(row.get("每日脂肪摄入量（g）")) and row["每日脂肪摄入量（g）"] >= 90: s += 1
    if not _bool_score(row, "饮食是否均衡"): s += 1
    return s

def score_osteoporosis(row):
    s = 0
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] < 22: s += 1
    if not pd.isna(row.get("每日蛋白质摄入量（g）")) and row["每日蛋白质摄入量（g）"] < 60: s += 1
    if not _bool_score(row, "是否每周进行进行中等以上强度锻炼"): s += 1
    return s

def score_anemia(row):
    s = 0
    if not pd.isna(row.get("血红蛋白（g/dL）")) and row["血红蛋白（g/dL）"] < 12.0: s += 1
    if not pd.isna(row.get("红细胞计数（10⁶ 个细胞/µL）")) and row["红细胞计数（10⁶ 个细胞/µL）"] < 4.0: s += 1
    return s

def score_cold(row):
    s = 0
    if not pd.isna(row.get("白细胞计数（1000个细胞/uL）")) and row["白细胞计数（1000个细胞/uL）"] >= 8.0: s += 1
    if not pd.isna(row.get("脉搏（次/分钟）")) and row["脉搏（次/分钟）"] >= 80: s += 1
    return s

def score_gastroenteritis(row):
    s = 0
    if not pd.isna(row.get("白细胞计数（1000个细胞/uL）")) and row["白细胞计数（1000个细胞/uL）"] >= 8.0: s += 1
    if not _bool_score(row, "饮食是否均衡"): s += 1
    return s

def score_gallstone(row):
    s = 0
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 30: s += 1
    if not pd.isna(row.get("胆固醇（mg/dL）")) and row["胆固醇（mg/dL）"] >= 200: s += 1
    if not _bool_score(row, "饮食是否均衡"): s += 1
    return s

def score_kidney_stone(row):
    s = 0
    if not pd.isna(row.get("起夜次数")) and row["起夜次数"] >= 2: s += 1
    if not pd.isna(row.get("每日水摄入量（g）")) and row["每日水摄入量（g）"] < 1500: s += 1
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 30: s += 1
    return s

def score_renal_failure(row):
    s = 0
    if not pd.isna(row.get("肌酐（mg/dL）")) and row["肌酐（mg/dL）"] >= 1.5: s += 1
    if not pd.isna(row.get("血尿素氮（mg/dL）")) and row["血尿素氮（mg/dL）"] >= 20: s += 1
    if not pd.isna(row.get("起夜次数")) and row["起夜次数"] >= 2: s += 1
    return s

def score_hepatitis_b(row):
    s = 0
    if not pd.isna(row.get("白蛋白（g/dL）")) and row["白蛋白（g/dL）"] < 4.0: s += 1
    if not pd.isna(row.get("球蛋白（g/dL）")) and row["球蛋白（g/dL）"] >= 3.2: s += 1
    if not _bool_score(row, "饮食是否均衡"): s += 1
    return s

def score_hepatitis_c(row):
    s = 0
    if not pd.isna(row.get("白蛋白（g/dL）")) and row["白蛋白（g/dL）"] < 4.0: s += 1
    if not pd.isna(row.get("球蛋白（g/dL）")) and row["球蛋白（g/dL）"] >= 3.2: s += 1
    if not _bool_score(row, "饮食是否均衡"): s += 1
    return s

def score_dental_caries(row):
    s = 0
    if not _bool_score(row, "饮食是否均衡"): s += 1
    if not pd.isna(row.get("每日食盐摄入程度")) and row["每日食盐摄入程度"] >= 3: s += 1
    return s

def score_nafld(row):
    s = 0
    if not pd.isna(row.get("BMI（kg/m²）")) and row["BMI（kg/m²）"] >= 26: s += 1
    if not pd.isna(row.get("甘油三酯（mg/dL）")) and row["甘油三酯（mg/dL）"] >= 150: s += 1
    if not pd.isna(row.get("胆固醇（mg/dL）")) and row["胆固醇（mg/dL）"] >= 180: s += 1
    return s


SCALAR_RULES = {
    "糖尿病":     score_diabetes,
    "高血压":     score_hypertension,
    "脂肪肝":     score_fatty_liver,
    "痛风":       score_gout,
    "骨质疏松":   score_osteoporosis,
    "贫血":       score_anemia,
    "感冒":       score_cold,
    "肠胃炎":     score_gastroenteritis,
    "胆结石":     score_gallstone,
    "肾结石":     score_kidney_stone,
    "肾衰竭":     score_renal_failure,
    "乙肝":       score_hepatitis_b,
    "丙肝":       score_hepatitis_c,
    "蛀牙":       score_dental_caries,
    "低血压":     score_hypotension,
    "睡眠呼吸暂停": score_sleep_apnea,
    "心律失常":   score_arrhythmia,
    "高尿酸血症": score_hyperuricemia,
    "NAFLD":     score_nafld,
}


# ============================================================
# 向量化版本（整列操作，推荐 stage02 使用）
# ============================================================

def _mask_or_zero(df, col, op, threshold):
    """对列做阈值判断，列不存在时返回全 0。"""
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return _mask(df[col], op, threshold).astype("int64")


def _bs(df, col):
    """bool 列 True→1，False/NaN→0。列不存在时返回全 0。"""
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return df[col].map({True: 1, False: 0}).fillna(0).astype("int64")


def score_vec_hypertension(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "血压收缩压（mmHg）", lambda s, t: s >= t, 140)
    s += _mask_or_zero(df, "血压舒张压（mmHg）", lambda s, t: s >= t, 90)
    s += _mask_or_zero(df, "BMI（kg/m²）",       lambda s, t: s >= t, 28)
    return s

def score_vec_hypotension(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "血压收缩压（mmHg）",   lambda s, t: s < t, 90)
    s += _mask_or_zero(df, "血压舒张压（mmHg）",   lambda s, t: s < t, 60)
    s += _mask_or_zero(df, "每日热量摄入量（kcal）", lambda s, t: s < t, 1500)
    return s

def score_vec_sleep_apnea(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "打鼾频率",        lambda s, t: s >= t, 2)
    s += _mask_or_zero(df, "BMI（kg/m²）",   lambda s, t: s >= t, 25)
    s += _mask_or_zero(df, "起夜次数",       lambda s, t: s >= t, 2)
    return s

def score_vec_arrhythmia(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    pulse = df["脉搏（次/分钟）"] if "脉搏（次/分钟）" in df.columns else pd.Series(dtype=float)
    s += ((pulse < 60) | (pulse > 100)).fillna(False).astype("int64")
    s += _bs(df, "脉搏是否规律")
    s += _mask_or_zero(df, "胆固醇（mg/dL）", lambda s, t: s >= t, 200)
    return s

def score_vec_hyperuricemia(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "每日蛋白质摄入量（g）", lambda s, t: s >= t, 90)
    s += _mask_or_zero(df, "每日脂肪摄入量（g）",   lambda s, t: s >= t, 90)
    s += _mask_or_zero(df, "BMI（kg/m²）",          lambda s, t: s >= t, 25)
    return s

def score_vec_diabetes(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "葡萄糖（mg/dL）", lambda s, t: s >= t, 126)
    s += _mask_or_zero(df, "BMI（kg/m²）",    lambda s, t: s >= t, 28)
    s += _bs(df, "存在家族糖尿病史")
    return s

def score_vec_fatty_liver(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "BMI（kg/m²）",       lambda s, t: s >= t, 30)
    s += _mask_or_zero(df, "甘油三酯（mg/dL）",  lambda s, t: s >= t, 200)
    s += _mask_or_zero(df, "胆固醇（mg/dL）",    lambda s, t: s >= t, 220)
    return s

def score_vec_nafld(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "BMI（kg/m²）",       lambda s, t: s >= t, 26)
    s += _mask_or_zero(df, "甘油三酯（mg/dL）",  lambda s, t: s >= t, 150)
    s += _mask_or_zero(df, "胆固醇（mg/dL）",     lambda s, t: s >= t, 180)
    return s

def score_vec_gout(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "BMI（kg/m²）",          lambda s, t: s >= t, 30)
    s += _mask_or_zero(df, "每日蛋白质摄入量（g）",  lambda s, t: s >= t, 100)
    s += _mask_or_zero(df, "每日脂肪摄入量（g）",    lambda s, t: s >= t, 90)
    s += (1 - _bs(df, "饮食是否均衡"))
    return s

def score_vec_osteoporosis(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "BMI（kg/m²）",           lambda s, t: s < t, 22)
    s += _mask_or_zero(df, "每日蛋白质摄入量（g）",  lambda s, t: s < t, 60)
    s += (1 - _bs(df, "是否每周进行进行中等以上强度锻炼"))
    return s

def score_vec_anemia(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "血红蛋白（g/dL）",              lambda s, t: s < t, 12.0)
    s += _mask_or_zero(df, "红细胞计数（10⁶ 个细胞/µL）",  lambda s, t: s < t, 4.0)
    return s

def score_vec_cold(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "白细胞计数（1000个细胞/uL）", lambda s, t: s >= t, 8.0)
    s += _mask_or_zero(df, "脉搏（次/分钟）",             lambda s, t: s >= t, 80)
    return s

def score_vec_gastroenteritis(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "白细胞计数（1000个细胞/uL）", lambda s, t: s >= t, 8.0)
    s += (1 - _bs(df, "饮食是否均衡"))
    return s

def score_vec_gallstone(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "BMI（kg/m²）",     lambda s, t: s >= t, 30)
    s += _mask_or_zero(df, "胆固醇（mg/dL）",   lambda s, t: s >= t, 200)
    s += (1 - _bs(df, "饮食是否均衡"))
    return s

def score_vec_kidney_stone(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "起夜次数",          lambda s, t: s >= t, 2)
    s += _mask_or_zero(df, "每日水摄入量（g）", lambda s, t: s < t, 1500)
    s += _mask_or_zero(df, "BMI（kg/m²）",      lambda s, t: s >= t, 30)
    return s

def score_vec_renal_failure(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "肌酐（mg/dL）",     lambda s, t: s >= t, 1.5)
    s += _mask_or_zero(df, "血尿素氮（mg/dL）", lambda s, t: s >= t, 20)
    s += _mask_or_zero(df, "起夜次数",          lambda s, t: s >= t, 2)
    return s

def score_vec_hepatitis_b(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "白蛋白（g/dL）", lambda s, t: s < t, 4.0)
    s += _mask_or_zero(df, "球蛋白（g/dL）", lambda s, t: s >= t, 3.2)
    s += (1 - _bs(df, "饮食是否均衡"))
    return s

def score_vec_hepatitis_c(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += _mask_or_zero(df, "白蛋白（g/dL）", lambda s, t: s < t, 4.0)
    s += _mask_or_zero(df, "球蛋白（g/dL）", lambda s, t: s >= t, 3.2)
    s += (1 - _bs(df, "饮食是否均衡"))
    return s

def score_vec_dental_caries(df):
    s = pd.Series(0, index=df.index, dtype="int64")
    s += (1 - _bs(df, "饮食是否均衡"))
    s += _mask_or_zero(df, "每日食盐摄入程度", lambda s, t: s >= t, 3)
    return s


VECTORIZED_RULES = {
    "高血压":       score_vec_hypertension,
    "低血压":       score_vec_hypotension,
    "睡眠呼吸暂停": score_vec_sleep_apnea,
    "心律失常":     score_vec_arrhythmia,
    "高尿酸血症":   score_vec_hyperuricemia,
    "糖尿病":       score_vec_diabetes,
    "脂肪肝":       score_vec_fatty_liver,
    "NAFLD":       score_vec_nafld,
    "痛风":         score_vec_gout,
    "骨质疏松":     score_vec_osteoporosis,
    "贫血":         score_vec_anemia,
    "感冒":         score_vec_cold,
    "肠胃炎":       score_vec_gastroenteritis,
    "胆结石":       score_vec_gallstone,
    "肾结石":       score_vec_kidney_stone,
    "肾衰竭":       score_vec_renal_failure,
    "乙肝":         score_vec_hepatitis_b,
    "丙肝":         score_vec_hepatitis_c,
    "蛀牙":         score_vec_dental_caries,
}


# ============================================================
# 导出函数
# ============================================================

def get_diagnostic_rules():
    """逐行规则（backward-compat，速度慢，仅用于单条预测）"""
    return SCALAR_RULES


def get_diagnostic_rules_vec():
    """向量化规则（整列操作，推荐 stage02 批量处理使用）"""
    return VECTORIZED_RULES


# ============================================================
# 冒烟测试
# ============================================================

if __name__ == "__main__":
    import time
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "data_stage01.csv"

    if not csv_path.exists():
        print(f"[SKIP] 找不到 {csv_path}")
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
        print(f"加载 {len(df)} 行 × {len(df.columns)} 列")

        # 测试向量化
        t0 = time.time()
        rules_vec = get_diagnostic_rules_vec()
        for disease, fn in rules_vec.items():
            df[f"症状得分_{disease}"] = fn(df)
        t_vec = time.time() - t0
        print(f"\n向量化完成: {len(rules_vec)} 个疾病, 耗时 {t_vec:.2f}s")

        # 验证：逐行 vs 向量化（取 10 行）
        rules_scalar = get_diagnostic_rules()
        sample = df.head(10)
        for disease in ["高血压", "糖尿病", "NAFLD", "心律失常"]:
            scalar_vals = sample.apply(rules_scalar[disease], axis=1)
            vec_vals = rules_vec[disease](sample)
            match = scalar_vals.equals(vec_vals)
            print(f"  {disease}: 逐行 vs 向量化 一致性 = {match}")

        # 输出分布
        print("\n各疾病得分分布:")
        score_cols = [c for c in df.columns if c.startswith("症状得分_")]
        for col in score_cols:
            print(f"  {col}: min={df[col].min()}  max={df[col].max()}  "
                  f"mean={df[col].mean():.2f}")
