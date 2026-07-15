from project.config import FEATURE_COLS_and_Default_VALUES

#此代码无用
'''
def convert_and_validate(data):
    """
    转换并校验前端数据
    输入：原始JSON数据（dict）
    输出：(是否成功, 转换后的数据dict 或 错误信息)
    """
    if not data:
        return False, None, "未收到数据"
    #=================数据格式转换=================#
    try:
        age = float(data.get('age', 0))
    except (TypeError, ValueError):
        return False, None, "年龄（age）格式错误，请输入数字"

    try:
        gender = int(data.get('gender', 0))
    except (TypeError, ValueError):
        return False, None, "性别（gender）格式错误，请输入 0（女）或 1（男）"

    try:
        systolic_bp = float(data.get('systolic_bp', 0))
    except (TypeError, ValueError):
        return False, None, "收缩压（systolic_bp）格式错误，请输入数字"

    try:
        diastolic_bp = float(data.get('diastolic_bp', 0))
    except (TypeError, ValueError):
        return False, None, "舒张压（diastolic_bp）格式错误，请输入数字"

    try:
        fasting_glucose = float(data.get('fasting_glucose', 0))
    except (TypeError, ValueError):
        return False, None, "空腹血糖（fasting_glucose）格式错误，请输入数字"
    try:
        height = float(data.get('height',0))
    except (TypeError, ValueError):
        return False, None, "身高（height）格式错误，请输入数字"

    try:
        weight = float(data.get('weight',0))
    except (TypeError, ValueError):
        return False, None, "体重（weight）格式错误，请输入数字"

    try:
        heart_rate = float(data.get('heart_rate',0))
    except (TypeError, ValueError):
        return False, None, "心率（heart_rate）格式错误，请输入数字"

    # 非必填字段，如果没有则取默认值


    try:
        total_cholesterol = float(data.get('total_cholesterol')) if data.get(
            'total_cholesterol') is not None else 4.5
    except (TypeError, ValueError):
        return False, None, "总胆固醇（total_cholesterol）格式错误，请输入数字"

    try:
        hdl_cholesterol = float(data.get('hdl_cholesterol')) if data.get('hdl_cholesterol') is not None else 1.3
    except (TypeError, ValueError):
        return False, None, "HDL胆固醇（hdl_cholesterol）格式错误，请输入数字"

    try:
        ldl_cholesterol = float(data.get('ldl_cholesterol')) if data.get('ldl_cholesterol') is not None else 2.8
    except (TypeError, ValueError):
        return False, None, "LDL胆固醇（ldl_cholesterol）格式错误，请输入数字"

    try:
        triglycerides = float(data.get('triglycerides')) if data.get('triglycerides') is not None else 1.5
    except (TypeError, ValueError):
        return False, None, "甘油三酯（triglycerides）格式错误，请输入数字"

    try:
        alt = float(data.get('alt')) if data.get('alt') is not None else 25.0
    except (TypeError, ValueError):
        return False, None, "ALT（alt）格式错误，请输入数字"

    try:
        ast = float(data.get('ast')) if data.get('ast') is not None else 22.0
    except (TypeError, ValueError):
        return False, None, "AST（ast）格式错误，请输入数字"

    try:
        creatinine = float(data.get('creatinine')) if data.get('creatinine') is not None else 80.0
    except (TypeError, ValueError):
        return False, None, "肌酐（creatinine）格式错误，请输入数字"

    try:
        urea = float(data.get('urea')) if data.get('urea') is not None else 5.5
    except (TypeError, ValueError):
        return False, None, "尿素（urea）格式错误，请输入数字"

    #=================数据校验=================#
    ###必填字段：
    if age < 0 or age > 150:
        return False, None, "年龄必须在 0-150 之间"

    if gender not in [0, 1]:
        return False, None, "性别必须为 0（女）或 1（男）"

    if systolic_bp < 50 or systolic_bp > 300:
        return False, None, "收缩压范围应在 50-300 mmHg 之间"

    if diastolic_bp < 30 or diastolic_bp > 200:
        return False, None, "舒张压范围应在 30-200 mmHg 之间"

    if fasting_glucose < 1.0 or fasting_glucose > 30.0:
        return False, None, "空腹血糖范围应在 1.0-30.0 mmol/L 之间"


    ###非必填字段：#如果字段存在且不为空则进行校验
    if 'height' in data and data['height'] is not None:
        if height < 50 or height > 300:
            return False, None, "身高范围应在 50-300 cm 之间"

    if 'weight' in data and data['weight'] is not None:
        if weight < 10 or weight > 500:
            return False, None, "体重范围应在 10-500 kg 之间"

    if 'heart_rate' in data and data['heart_rate'] is not None:
        if heart_rate < 20 or heart_rate > 250:
            return False, None, "心率范围应在 20-250 次/分之间"

    if 'total_cholesterol' in data and data['total_cholesterol'] is not None:
        if total_cholesterol < 0.5 or total_cholesterol > 20.0:
            return False, None, "总胆固醇范围应在 0.5-20.0 mmol/L 之间"

    if 'hdl_cholesterol' in data and data['hdl_cholesterol'] is not None:
        if hdl_cholesterol < 0.1 or hdl_cholesterol > 5.0:
            return False, None, "HDL胆固醇范围应在 0.1-5.0 mmol/L 之间"

    if 'ldl_cholesterol' in data and data['ldl_cholesterol'] is not None:
        if ldl_cholesterol < 0.1 or ldl_cholesterol > 10.0:
            return False, None, "LDL胆固醇范围应在 0.1-10.0 mmol/L 之间"

    if 'triglycerides' in data and data['triglycerides'] is not None:
        if triglycerides < 0.1 or triglycerides > 20.0:
            return False, None, "甘油三酯范围应在 0.1-20.0 mmol/L 之间"

    if 'alt' in data and data['alt'] is not None:
        if alt < 1 or alt > 500:
            return False, None, "ALT范围应在 1-500 U/L 之间"

    if 'ast' in data and data['ast'] is not None:
        if ast < 1 or ast > 500:
            return False, None, "AST范围应在 1-500 U/L 之间"

    if 'creatinine' in data and data['creatinine'] is not None:
        if creatinine < 10 or creatinine > 2000:
            return False, None, "肌酐范围应在 10-2000 μmol/L 之间"

    if 'urea' in data and data['urea'] is not None:
        if urea < 0.5 or urea > 50.0:
            return False, None, "尿素范围应在 0.5-50.0 mmol/L 之间"

    converted = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'fasting_glucose': fasting_glucose,
        'total_cholesterol': total_cholesterol,
        'hdl_cholesterol': hdl_cholesterol,
        'ldl_cholesterol': ldl_cholesterol,
        'triglycerides': triglycerides,
        'alt': alt,
        'ast': ast,
        'creatinine': creatinine,
        'urea': urea,
        'symptom_headache': int(bool(data.get('symptom_headache', 0))),
        'symptom_dizziness': int(bool(data.get('symptom_dizziness', 0))),
        'symptom_chest_pain': int(bool(data.get('symptom_chest_pain', 0))),
        'symptom_palpitations': int(bool(data.get('symptom_palpitations', 0))),
        'symptom_shortness_breath': int(bool(data.get('symptom_shortness_breath', 0))),
        'symptom_fatigue': int(bool(data.get('symptom_fatigue', 0))),
    }       #转换并校验后的数据

    return True, converted, None
'''
def convert_and_validate(data:dict)->dict:
    # 转换并校验前端数据
    # 根据 config 校验并填充缺失值
    type_map = {
        "int": int,
        "float": float,
    "bool": lambda x: bool(x) if isinstance(x,bool) else str(x).lower() in ('true', '1', 'yes'),
        "str": str
    }

    result = {}
    for key, field_config in FEATURE_COLS_and_Default_VALUES.items():
        if key == "患病情况":
            result[key] = field_config.get("default", "")
            continue
        value = data.get(key)
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
#仅在模型需要列表时使用，已废弃
'''
def build_feature_vector(data:dict)->list:
    features = []
    for col, field_config in FEATURE_COLS_and_Default_VALUES.items():  # ← 使用config中定义的顺序
        if col == "BMI（kg/m²）":
            height = data.get('身高（cm）')
            weight = data.get('体重（kg）')
            value = weight / ((height / 100) ** 2)
        else:
            value = data.get(col, field_config.get('default', 0))
        features.append(float(value))
    return [features]
'''
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
def do_prediction(data:dict)->dict:
    pass
    '''
    disease = function(data)
    return disease
    '''