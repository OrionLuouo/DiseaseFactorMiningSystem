"""
demo_predict_flow.py — 演示"前端 → app.py → ML"完整数据流
===========================================================
模拟前端 1.html 表单提交，从 app.py 的字段映射 → stage04.predict() → 返回 Top-3。

运行方式：
    python ml/demo_predict_flow.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 把项目根目录加入路径
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("demo")


# ============================================================
# 第一步：模拟前端表单提交（蛇形命名字段，与 1.html 对应）
# ============================================================

FRONTEND_FORM_DATA = {
    # 基本信息
    "gender": "1",            # 1=男, 0=女
    # 生命体征
    "sbp": 162,
    "dbp": 100,
    "pulse": 82,
    # 体格
    "height": 175,
    "weight": 82,
    "bmi": 26.8,
    # 血常规
    "cholesterol": 240,
    "white_blood_cells": 7.5,
    "red_blood_cells": 5.1,
    "hemoglobin": 14.2,
    "platelets": 230,
    # 生化
    "glucose": 105,
    "triglycerides": 195,
    "creatinine": 1.05,
    "phosphorus": 3.6,
    "potassium": 4.1,
    "albumin": 4.3,
    "blood_urea_nitrogen": 16.0,
    "bicarbonate": 24.5,
    "sodium": 140.0,
    "globulin": 2.8,
    # 饮食
    "salt_intake": 3,
    "daily_calories": 2200,
    "daily_protein": 80,
    "daily_carbs": 280,
    "daily_fiber": 18,
    "daily_fat": 80,
    "daily_water": 1600,
    # 生活习惯
    "alcohol_freq": 2,
    "night_urination": 1,
    "sitting_hours": 360,
    "sleep_hours": 5.5,
    "snoring_freq": 3,
    # 布尔（是否勾选）
    "diet_sufficient": True,
    "family_diabetes": False,
    "diet_balanced": True,
    "physical_labor": False,
    "exercise": False,
    "pulse_regular": True,
    # 病史（前端用英文 key，与 HISTORY_FIELD_MAP 对应）
    "history": {
        "hypertension": 1,  # 声称有高血压史
        "fatty_liver": 0,
        "diabetes": 0,
    },
}


# ============================================================
# 第二步：模拟 app.py 的字段映射层（与 stage04.FRONTEND_TO_MODEL 对齐）
# ============================================================

def _convert_frontend_to_frontend_dict(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    把前端表单的「蛇形 key」转成 stage04 期望的「蛇形 key」
    实际这里不做字段名转换（FRONTEND_TO_MODEL 已经在 stage04 里做了），
    这里只做类型转换（gender "0"/"1" → "男"/"女"，bool 字段确认）。
    """
    result = dict(form_data)

    # gender: "0"/"1" → "男"/"女"
    g = result.get("gender")
    if g in ("0", "1"):
        result["gender"] = "男" if g == "1" else "女"

    return result


# ============================================================
# 第三步：调用 stage04.predict()，返回 Top-3
# ============================================================

def run_demo() -> Dict[str, Any]:
    from ml.stage04_predict import (
        predict,
        FRONTEND_TO_MODEL,
        HISTORY_FIELD_MAP,
        NORMAL_REFERENCE,
    )

    print("\n" + "=" * 70)
    print("  STAGE 1 — 模拟前端 1.html 表单提交")
    print("=" * 70)
    print(f"前端表单共 {len(FRONTEND_FORM_DATA)} 个字段（含 history 字典）")
    print(f"病史字段 keys: {list(FRONTEND_FORM_DATA['history'].keys())}")
    print("\n原始前端 JSON（前 5 个字段预览）:")
    preview = {k: FRONTEND_FORM_DATA[k] for k in list(FRONTEND_FORM_DATA.keys())[:5]}
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("  STAGE 2 — app.py 字段映射（前端 → 模型特征名）")
    print("=" * 70)
    frontend_dict = _convert_frontend_to_frontend_dict(FRONTEND_FORM_DATA)

    # 演示 FRONTEND_TO_MODEL 映射
    mapped_sample = {}
    for fe_key, model_col in list(FRONTEND_TO_MODEL.items())[:8]:
        val = frontend_dict.get(fe_key)
        mapped_sample[model_col] = val
    print(f"FRONTEND_TO_MODEL 映射（共 {len(FRONTEND_TO_MODEL)} 个字段）")
    print("示例（前 8 个字段）:")
    for k, v in mapped_sample.items():
        print(f"  {k:<25} ← {v}")

    print("\n" + "=" * 70)
    print("  STAGE 3 — 调用 stage04.predict()")
    print("=" * 70)
    history = frontend_dict.pop("history", None)
    print(f"history 字典: {history}")

    # 调用 predict
    result = predict(
        frontend_dict,
        history=history,
        top_n=3,
    )

    print("\n" + "=" * 70)
    print("  STAGE 4 — 返回结果（最终给前端）")
    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    # 把 PowerShell 控制台编码乱码问题绕开：用 logging + 文件输出
    log_path = Path(__file__).resolve().parent.parent / "data" / "intermediate" / "demo_predict_flow.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    result = run_demo()

    print(f"\n完整输出已写入: {log_path}")
    print(f"Top 疾病: {result['top_disease']} (置信度 {result['confidence']:.2%})")