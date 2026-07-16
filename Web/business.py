from project.config import FEATURE_COLS_and_Default_VALUES
from ml.stage04_predict import predict

def convert_and_validate(data:dict)->dict:
    # 转换并校验前端数据
    # 根据 config 校验并填充缺失值
    type_map = {
        "int": int,
        "float": float,
        "bool": lambda x: bool(x) if isinstance(x,bool) else str(x).lower() in ('true', '1', 'yes'),
        "str": str
    }
    data_map = data['data']

    result = {}
    for key, field_config in FEATURE_COLS_and_Default_VALUES.items():
        if key == "患病情况":
            result[key] = field_config.get("default", "")
            continue
        value = data_map[key]
        min_value = field_config.get("min")  # 如果没有配置，返回 None
        max_value = field_config.get("max")  # 如果没有配置，返回 None

        if value is None or value == "":
            # 如果字段是必要字段且缺失，报错
            if field_config.get("necessary", False):
                raise ValueError(f"必要字段 '{key}' 缺失")
            # 否则使用默认值
            value = field_config["default"]

        try:            # 数据类型转换（char->int/float/bool）
            converter = type_map.get(field_config["type"])
            if converter:
                value = converter(value)
        except (ValueError, TypeError):
            raise ValueError(f"字段 '{key}' 类型错误，期望 {field_config['type']}，\
            实际 {type(value).__name__}")


            #检查各项值是否在可接受范围内
        if min_value is not None and value < min_value:
            raise ValueError(f"'{key}'值低于最低检测'{min_value}'")
        if max_value is not None and value > max_value:
            raise ValueError(f"'{key}'值高于最高检测'{max_value}'")

        result[key] = value
    height = result.get('身高（cm）')
    weight = result.get('体重（kg）')
    bmi = weight / ((height / 100) ** 2)
    result['BMI（kg/m²）'] = round(bmi,2)
    return result

#此段代码仅作测试用
'''
def do_prediction_demo(data):
    """根据用户数据生成预测结果（模拟版）"""
    sbp = data.get('systolic_bp', 120)
    dbp = data.get('diastolic_bp', 80)
    glucose = data.get('fasting_glucose', 5.6)

    # 模拟Top3疾病
    top_diseases = []

    # 高血压判断
    if sbp >= 140 or dbp >= 90:
        top_diseases.append({"disease": "高血压", "probability": 0.85})
    else:
        top_diseases.append({"disease": "高血压", "probability": 0.15})

    # 糖尿病判断
    if glucose >= 7.0:
        top_diseases.append({"disease": "2型糖尿病", "probability": 0.78})
    elif glucose >= 6.1:
        top_diseases.append({"disease": "2型糖尿病", "probability": 0.52})
    else:
        top_diseases.append({"disease": "2型糖尿病", "probability": 0.12})

    # 冠心病判断（模拟）
    if sbp >= 160 or glucose >= 8.0:
        top_diseases.append({"disease": "冠心病", "probability": 0.62})
    else:
        top_diseases.append({"disease": "冠心病", "probability": 0.18})

    # 按概率从高到低排序，取Top3
    top_diseases.sort(key=lambda x: x['probability'], reverse=True)
    top3 = top_diseases[:3]

    # 主要预测疾病 = Top1
    predicted_disease = top3[0]['disease']
    confidence = top3[0]['probability']

    # 健康状态判断
    if confidence >= 0.6:
        health_status = "患病"
    elif confidence >= 0.3:
        health_status = "建议复查"
    else:
        health_status = "基本健康"

    # 指标对比（模拟正常范围）
    indicator_comparison = {
        "systolic_bp": {
            "value": sbp,
            "normal_range": [105, 145],
            "status": "偏高" if sbp > 145 else ("偏低" if sbp < 105 else "正常"),
            "deviation": round((sbp - 125) / 15, 2)
        },
        "diastolic_bp": {
            "value": dbp,
            "normal_range": [65, 95],
            "status": "偏高" if dbp > 95 else ("偏低" if dbp < 65 else "正常"),
            "deviation": round((dbp - 80) / 10, 2)
        },
        "fasting_glucose": {
            "value": glucose,
            "normal_range": [4.0, 7.0],
            "status": "偏高" if glucose > 7.0 else ("偏低" if glucose < 4.0 else "正常"),
            "deviation": round((glucose - 5.5) / 1.2, 2)
        }
    }

    # 雷达图数据
    radar_data = {
        "indicators": ["血压", "血糖", "BMI", "运动", "饮食", "睡眠"],
        "scores": [65, 55, 70, 50, 60, 75]
    }

    # 健康建议
    recommendations = []
    if sbp > 145:
        recommendations.append("收缩压偏高，建议限盐饮食，每周测量血压")
    if glucose > 7.0:
        recommendations.append("空腹血糖偏高，建议控制碳水化合物摄入，定期监测血糖")
    if sbp < 105:
        recommendations.append("收缩压偏低，建议适当增加营养，避免体位突变")
    if not recommendations:
        recommendations.append("各项指标基本正常，建议保持健康生活方式")

    return {
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "top_possible_diseases": top3,
        "health_status": health_status,
        "indicator_comparison": indicator_comparison,
        "radar_data": radar_data,
        "recommendations": recommendations
    }
'''

def _get_score(value, target, tolerance):
    """计算单项健康得分（0-100）"""
    if value <= 0:
        return 50
    diff = abs(value - target) / tolerance if tolerance > 0 else 0
    score = max(0, min(100, 100 - diff * 20))
    return round(score, 0)

def do_prediction(data:dict)->dict:
    result = predict(data,top_n=3)
    return result