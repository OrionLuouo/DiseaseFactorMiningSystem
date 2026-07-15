# stage04 预测接口使用说明 / API Usage Guide

> **`ml/stage04_predict.py` — 新病人预测接口**
> 输入前端表单 → 输出 19 种疾病概率 Top-N。

---

## 一、概述 / Overview

| 项目 / Item | 内容 / Content |
|---|---|
| 模块 / Module | `ml/stage04_predict.py` |
| 主函数 / Main Function | `predict(user_dict, history=None, top_n=3)` |
| 批量函数 / Batch Function | `predict_batch(user_dicts, history_list=None, top_n=3)` |
| 辅助函数 / Helper | `get_input_schema()` |
| 模型 / Model | `ml/models/xgboost_chain_v1.joblib` |
| 输入特征数 / Input Features | 40 个体检指标 + 19 个症状得分 = 59 列 |
| 输出标签数 / Output Labels | 19 个疾病标签 |
| 训练框架 / Framework | XGBoost + Classifier Chain（链式分类） |

---

## 二、快速调用 / Quick Start

### 2.1 Python 调用 / Python Call

```python
from ml.stage04_predict import predict

result = predict(
    user_dict={
        "gender": "男",
        "sbp": 162,
        "dbp": 100,
        "pulse": 82,
        "weight": 82,
        "height": 175,
        "bmi": 26.8,
        "cholesterol": 240,
        "white_blood_cells": 7.5,
        # ... 其余字段见下表
    },
    history={"hypertension": 1},   # 可选
    top_n=3,                        # 返回 Top-3
)

print(result["top_disease"])       # → "患病情况_肾结石"
print(result["confidence"])        # → 0.1583
print(result["top_n"])             # → Top-3 列表
```

### 2.2 批量调用 / Batch Call

```python
from ml.stage04_predict import predict_batch

results = predict_batch(
    user_dicts=[user_dict_1, user_dict_2, user_dict_3],
    history_list=[{"hypertension": 1}, None, {"diabetes": 1}],
    top_n=3,
)
# 返回值: list[dict], 长度 = len(user_dicts)
```

---

## 三、API 函数签名 / API Signatures

### 3.1 `predict()` — 单次预测

```python
def predict(
    user_dict: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    model_path: Optional[Union[str, Path]] = None,
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    top_n: int = 3,
) -> Dict[str, Any]:
```

| 参数 / Parameter | 类型 / Type | 必需 / Required | 默认 / Default | 说明 / Description |
|---|---|---|---|---|
| `user_dict` | `dict` | ✅ 是 | — | 前端表单数据，键名见下表 |
| `history` | `dict` | ❌ 否 | `None` | 病史字典，支持英文 key 或中文病名 |
| `model_path` | `str`/`Path` | ❌ 否 | `models/xgboost_chain_v1.joblib` | 模型路径 |
| `norm_stats` | `dict` | ❌ 否 | 自动加载 | 归一化统计量（一般不用传） |
| `top_n` | `int` | ❌ 否 | `3` | 返回 Top-N 疾病（1–19） |

### 3.2 `predict_batch()` — 批量预测

```python
def predict_batch(
    user_dicts: List[Dict[str, Any]],
    history_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    model_path: Optional[Union[str, Path]] = None,
    norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
```

| 参数 / Parameter | 类型 / Type | 必需 / Required | 默认 / Default | 说明 / Description |
|---|---|---|---|---|
| `user_dicts` | `list[dict]` | ✅ 是 | — | 用户字典列表 |
| `history_list` | `list[dict]` | ❌ 否 | 全为 `None` | 每个用户的病史，与 `user_dicts` 等长 |
| 其余参数 | 同 `predict()` | — | — | — |

### 3.3 `get_input_schema()` — 字段 schema

```python
def get_input_schema() -> Dict[str, Any]:
```

返回前端字段定义（用于动态生成表单）。**返回值结构**：

```python
{
    "fields": {
        "sbp": {"type": "number",   "model_field": "血压收缩压（mmHg）"},
        "exercise": {"type": "boolean", "model_field": "是否每周进行进行中等以上强度锻炼"},
        "gender": {"type": "string", "options": ["男", "女"], "model_field": "性别"},
        # ... 共 40 项
    },
    "history_fields": {
        "hypertension": {"model_field": "高血压"},
        "diabetes":     {"model_field": "糖尿病"},
        # ... 共 19 项
    },
}
```

---

## 四、输入字段 / Input Fields

`user_dict` 共 **40 个字段**，分为 6 类。

### 4.1 基本信息 / Basic Info

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 范围 / Range | 说明 / Description |
|---|---|---|---|---|
| `gender` | 性别 | `string` | `"男"` / `"女"` / `"M"` / `"F"` / `"1"` / `"0"` | 性别 / Gender |

### 4.2 生命体征 / Vital Signs

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 单位 / Unit | 说明 / Description |
|---|---|---|---|---|
| `sbp` | 血压收缩压（mmHg） | `number` | mmHg | 收缩压 / Systolic BP |
| `dbp` | 血压舒张压（mmHg） | `number` | mmHg | 舒张压 / Diastolic BP |
| `pulse` | 脉搏（次/分钟） | `number` | 次/分 | 心率 / Heart Rate |

### 4.3 体格指标 / Body Measurements

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 单位 / Unit | 说明 / Description |
|---|---|---|---|---|
| `weight` | 体重（kg） | `number` | kg | 体重 / Body Weight |
| `height` | 身高（cm） | `number` | cm | 身高 / Body Height |
| `bmi` | BMI（kg/m²） | `number` | kg/m² | 体质指数 / Body Mass Index |

### 4.4 血液检查 / Blood Test

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 单位 / Unit | 说明 / Description |
|---|---|---|---|---|
| `cholesterol` | 胆固醇（mg/dL） | `number` | mg/dL | 总胆固醇 / Cholesterol |
| `white_blood_cells` | 白细胞计数（1000个细胞/uL） | `number` | 10³/µL | 白细胞 / WBC |
| `red_blood_cells` | 红细胞计数（10⁶ 个细胞/µL） | `number` | 10⁶/µL | 红细胞 / RBC |
| `hemoglobin` | 血红蛋白（g/dL） | `number` | g/dL | 血红蛋白 / Hemoglobin |
| `platelets` | 血小板计数（1000 个细胞/µL） | `number` | 10³/µL | 血小板 / Platelets |

### 4.5 生化指标 / Biochemistry

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 单位 / Unit | 说明 / Description |
|---|---|---|---|---|
| `glucose` | 葡萄糖（mg/dL） | `number` | mg/dL | 血糖 / Glucose |
| `triglycerides` | 甘油三酯（mg/dL） | `number` | mg/dL | 甘油三酯 / Triglycerides |
| `creatinine` | 肌酐（mg/dL） | `number` | mg/dL | 肌酐 / Creatinine |
| `phosphorus` | 磷（mg/dL） | `number` | mg/dL | 血磷 / Phosphorus |
| `potassium` | 钾（mmol/L） | `number` | mmol/L | 血钾 / Potassium |
| `albumin` | 白蛋白（g/dL） | `number` | g/dL | 白蛋白 / Albumin |
| `blood_urea_nitrogen` | 血尿素氮（mg/dL） | `number` | mg/dL | 尿素氮 / BUN |
| `bicarbonate` | 碳酸氢盐（mmol/L） | `number` | mmol/L | 碳酸氢盐 / Bicarbonate |
| `sodium` | 钠（mmol/L） | `number` | mmol/L | 血钠 / Sodium |
| `globulin` | 球蛋白（g/dL） | `number` | g/dL | 球蛋白 / Globulin |

### 4.6 饮食与生活习惯 / Diet & Lifestyle

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 单位 / Unit | 说明 / Description |
|---|---|---|---|---|
| `salt_intake` | 每日食盐摄入程度 | `number` | 1–5 | 盐摄入等级 / Salt Intake Level |
| `daily_calories` | 每日热量摄入量（kcal） | `number` | kcal | 热量 / Daily Calories |
| `daily_protein` | 每日蛋白质摄入量（g） | `number` | g | 蛋白质 / Protein |
| `daily_carbs` | 每日碳水摄入量（g） | `number` | g | 碳水 / Carbohydrates |
| `daily_fiber` | 每日膳食纤维摄入量（g） | `number` | g | 膳食纤维 / Fiber |
| `daily_fat` | 每日脂肪摄入量（g） | `number` | g | 脂肪 / Fat |
| `daily_water` | 每日水摄入量（g） | `number` | g | 饮水量 / Water |
| `alcohol_freq` | 饮酒频率 | `number` | 0–5 | 频率等级 / Alcohol Frequency |
| `night_urination` | 起夜次数 | `number` | 次 | 起夜次数 / Nocturia |
| `sitting_hours` | 平均每天坐姿时长（分钟） | `number` | 分钟 | 久坐时长 / Sitting Hours |
| `sleep_hours` | 工作日平均睡眠小时数 | `number` | 小时 | 睡眠时长 / Sleep Hours |
| `snoring_freq` | 打鼾频率 | `number` | 0–5 | 打鼾等级 / Snoring Frequency |

### 4.7 布尔字段 / Boolean Fields

| 前端键 / Frontend Key | 模型列 / Model Field | 类型 / Type | 说明 / Description |
|---|---|---|---|
| `diet_sufficient` | 饮食是否充足 | `boolean` | 饮食充足 / Diet Sufficient |
| `family_diabetes` | 存在家族糖尿病史 | `boolean` | 家族糖尿病史 / Family Diabetes History |
| `diet_balanced` | 饮食是否均衡 | `boolean` | 饮食均衡 / Balanced Diet |
| `physical_labor` | 是否从事体力劳动 | `boolean` | 体力劳动 / Physical Labor |
| `exercise` | 是否每周进行进行中等以上强度锻炼 | `boolean` | 每周中等强度运动 / Weekly Exercise |
| `pulse_regular` | 脉搏是否规律 | `boolean` | 脉搏规律 / Regular Pulse |

> **布尔值接受 / Boolean Accepted**：`True`/`False`、`"true"`/`"false"`、`"1"`/`"0"`、`"是"`/`"否"`、数字 `1`/`0`。

---

## 五、病史字段 / History Field

`history` 字典共 **19 个键**，与 19 个疾病一一对应。

> **病史只能"升高"概率，不能降低；与症状矛盾的病史被忽略并产生警告。**

### 5.1 支持的病史字段 / Supported History Keys

| 前端键 / Frontend Key | 映射病名 / Disease | 中文 / Chinese |
|---|---|---|
| `hypertension` | 高血压 | Hypertension |
| `hypotension` | 低血压 | Hypotension |
| `sleep_apnea` | 睡眠呼吸暂停 | Sleep Apnea |
| `arrhythmia` | 心律失常 | Arrhythmia |
| `hyperuricemia` | 高尿酸血症 | Hyperuricemia |
| `diabetes` | 糖尿病 | Diabetes |
| `fatty_liver` | 脂肪肝 | Fatty Liver |
| `nafld` | NAFLD | Non-alcoholic Fatty Liver Disease |
| `gout` | 痛风 | Gout |
| `osteoporosis` | 骨质疏松 | Osteoporosis |
| `anemia` | 贫血 | Anemia |
| `common_cold` | 感冒 | Common Cold |
| `gastroenteritis` | 肠胃炎 | Gastroenteritis |
| `gallstones` | 胆结石 | Gallstones |
| `kidney_stones` | 肾结石 | Kidney Stones |
| `kidney_failure` | 肾衰竭 | Kidney Failure |
| `hepatitis_b` | 乙肝 | Hepatitis B |
| `hepatitis_c` | 丙肝 | Hepatitis C |
| `cavities` | 蛀牙 | Cavities |

### 5.2 病史值 / History Values

```python
# 两种写法都支持 / Both formats supported:
history = {"hypertension": 1, "diabetes": 0}                      # 英文 key / English keys
history = {"高血压": 1, "糖尿病": 0}                              # 中文病名 / Chinese names
history = {"hypertension": True, "fatty_liver": False}            # 布尔值 / Booleans
```

---

## 六、返回值 / Return Value

`predict()` 返回一个 **dict**，包含 5 个字段：

### 6.1 返回值结构 / Return Schema

```json
{
    "top_disease": "患病情况_NAFLD",
    "confidence": 0.8670,
    "top_n": [
        {"rank": 1, "disease": "患病情况_NAFLD",   "probability": 0.8670},
        {"rank": 2, "disease": "患病情况_肾结石",   "probability": 0.0234},
        {"rank": 3, "disease": "患病情况_蛀牙",     "probability": 0.0195}
    ],
    "warnings": [],
    "normal_reference": {
        "性别": 1,
        "血压收缩压（mmHg）": 120.0,
        "BMI（kg/m²）": 22.0,
        "...": "..."
    }
}
```

### 6.2 字段说明 / Field Description

| 字段 / Field | 类型 / Type | 说明 / Description |
|---|---|---|
| `top_disease` | `str` | 模型预测的最可能疾病（前缀 `患病情况_`，用于显示请去除前缀）<br/>Most likely disease (strip `患病情况_` prefix for display) |
| `confidence` | `float` | 最可能疾病的概率（0–1）<br/>Probability of `top_disease` (0–1) |
| `top_n` | `list[dict]` | Top-N 疾病列表，按概率降序<br/>Top-N diseases sorted by probability |
| `warnings` | `list[str]` | 病史冲突警告列表，详见 §7<br/>History conflict warnings (see §7) |
| `normal_reference` | `dict[str, float]` | 正常人参考值（40 项），用于前端雷达图 / 对比表<br/>Normal reference values for radar chart |

### 6.3 `top_n` 单条结构 / Single Item Schema

```json
{
    "rank": 1,              // 排名 / Rank (1-based)
    "disease": "患病情况_NAFLD",  // 疾病名（含 "患病情况_" 前缀）
    "probability": 0.8670   // 概率 / Probability (0–1)
}
```

---

## 七、病史校验机制 / History Validation

### 7.1 校验规则 / Validation Rule

| 模型概率区间 / Probability Range | 契合度 / Fitness | 提升系数 / Boost | 行为 / Behavior |
|---|---|---|---|
| `p ≥ threshold` | high | 1.0 | 概率升高 30% × (1 − p) |
| `threshold/2 ≤ p < threshold` | medium | 0.5 | 概率升高 15% × (1 − p) |
| `p < threshold/2` | low | 0.0 | **病史被忽略**，产生警告 |

> `threshold` 是模型在训练时为每个标签搜索的最优决策阈值（见 `stage03_evaluation_report.json`）。

### 7.2 警告样例 / Warning Examples

```python
# 历史声称有"高血压"，但症状指标不支持
warnings = [
    "「高血压」：症状指标不支持您声称的病史（模型概率 2.5%），"
    "建议到院复查。该病史未纳入本次预测。"
]
```

---

## 八、错误处理 / Error Handling

### 8.1 异常情况 / Exceptions

| 异常 / Exception | 触发条件 / Trigger | 解决方法 / Solution |
|---|---|---|
| `FileNotFoundError` | 模型文件不存在 | 先运行 `python ml/stage03_ML_train.py` |
| `FileNotFoundError` | 归一化统计量文件不存在 | 先运行 `python ml/stage02_preprocess.py` |
| `ValueError` | 模型包缺少必要字段 | 重新训练模型 |
| `ValueError` | `top_n < 1` 或 `top_n > 19` | 调整 `top_n` 参数 |

### 8.2 推荐调用包装 / Recommended Wrapper

```python
from ml.stage04_predict import predict

def safe_predict(user_dict, history=None, top_n=3):
    try:
        return {"ok": True, "data": predict(user_dict, history, top_n=top_n)}
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e), "type": "ModelNotFound"}
    except ValueError as e:
        return {"ok": False, "error": str(e), "type": "ValidationError"}
    except Exception as e:
        return {"ok": False, "error": str(e), "type": type(e).__name__}
```

---

## 九、完整示例 / Complete Example

### 9.1 单次预测 / Single Prediction

```python
from ml.stage04_predict import predict

user_dict = {
    # 基本信息
    "gender": "男",

    # 生命体征
    "sbp": 162,
    "dbp": 100,
    "pulse": 82,

    # 体格
    "weight": 82,
    "height": 175,
    "bmi": 26.8,

    # 血液检查
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

    # 布尔
    "diet_sufficient": True,
    "family_diabetes": False,
    "diet_balanced": True,
    "physical_labor": False,
    "exercise": False,
    "pulse_regular": True,
}

history = {"hypertension": 1, "fatty_liver": 0, "diabetes": 0}

result = predict(user_dict, history=history, top_n=3)

print(f"最可能疾病: {result['top_disease']}")
print(f"置信度: {result['confidence']:.2%}")
for item in result["top_n"]:
    print(f"  #{item['rank']} {item['disease']:<25} {item['probability']:.4f}")
```

**输出示例 / Sample Output**：

```
最可能疾病: 患病情况_肾结石
置信度: 15.83%
  #1 患病情况_肾结石               0.1583
  #2 患病情况_胆结石               0.0516
  #3 患病情况_蛀牙                 0.0038
```

### 9.2 Flask 集成示例 / Flask Integration

```python
from flask import Flask, request, jsonify
from ml.stage04_predict import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    payload = request.get_json(force=True) or {}
    user_dict = payload.get("data", {})
    history   = payload.get("history", {})
    top_n     = int(payload.get("top_n", 3))

    try:
        result = predict(user_dict, history=history, top_n=top_n)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
```

### 9.3 命令行测试 / CLI Test

```bash
# 运行自带的演示
python ml/demo_predict_flow.py

# 或直接运行 stage04 自带的测试入口
python ml/stage04_predict.py
```

---

## 十、注意事项 / Caveats

> ⚠️ **生产部署前请确认 / Before Production Deployment**

1. **必需的训练产物 / Required Artifacts**
   - `ml/models/xgboost_chain_v1.joblib` — 模型包
   - `data/intermediate/stage02_norm_stats.json` — 归一化统计量

2. **环境变量 / Environment Variables**
   - `MODEL_PATH` — 模型路径（可选，默认自动找）

3. **字段缺失 / Missing Fields**
   - `user_dict` 字段可缺失：缺失列自动填 0（影响模型预测）
   - 建议前端尽量提供完整 40 项体检数据

4. **预测速度 / Latency**
   - 首次调用约 **0.5–2 秒**（加载模型）；之后约 **0.05–0.2 秒**（全局缓存）

5. **并发 / Concurrency**
   - `_cached_model` 是全局单例；多线程/多进程调用安全
   - **注意**：每个 Python 进程独立缓存，多 worker 部署会重复加载

6. **诊断规则 / Diagnostic Rules**
   - 症状得分依赖 `ml/diagnostic_rules.py`
   - 修改诊断规则后必须重新训练模型（`stage03`）

---

## 十一、相关文件 / Related Files

| 路径 / Path | 作用 / Purpose |
|---|---|
| `ml/stage04_predict.py` | 本接口实现 / This API |
| `ml/stage03_ML_train.py` | 模型训练 / Model Training |
| `ml/stage02_preprocess.py` | 预处理 + 归一化统计量 / Preprocessing |
| `ml/stage01_load.py` | 数据加载 / Data Loading |
| `ml/diagnostic_rules.py` | 症状得分规则 / Symptom Scoring Rules |
| `ml/demo_predict_flow.py` | 数据流演示 / Demo Script |
| `model_documentation.md` | 模型详细文档 / Model Documentation |

---

*文档版本 / Doc Version*: v1.0 · 配套 stage04_predict.py (737 行)
