

def check_data(data):      #校验数据函数
    if not data:
        return False,'未收到数据'

    required_fields = ['age','gender','systolic_bp','diastolic_bp','fasting_glucose']
    for field in required_fields:
        if field not in data or data.get(field) is None:
            return False,'缺少必要字段:'+field
        # 年龄范围检查
    age = data.get('age')
    if not isinstance(age, (int, float)) or age < 0 or age > 150:
        return False, "年龄必须在0-150之间"

    # 性别检查
    gender = data.get('gender')
    if gender not in [0, 1]:
        return False, "性别必须为0（女）或1（男）"

    # 收缩压范围检查
    sbp = data.get('systolic_bp')
    if sbp < 50 or sbp > 300:
        return False, "收缩压范围应在50-300之间"

    # 舒张压范围检查
    dbp = data.get('diastolic_bp')
    if dbp < 30 or dbp > 200:
        return False, "舒张压范围应在30-200之间"

    # 空腹血糖范围检查
    glucose = data.get('fasting_glucose')
    if glucose < 1.0 or glucose > 30.0:
        return False, "空腹血糖范围应在1.0-30.0之间"

    return True,'数据校验通过'

def do_prediction(data):
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