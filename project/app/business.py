from project.config import FEATURE_COLS_and_Default_VALUES

#此代码无用
'''
def validate_data(data):
    """
    校验输入数据的完整性和格式正确性
    返回: (is_valid, cleaned_data, error_message)
    """
    cleaned = {}

    # ==================== 疾病史 ====================
    # disease 为文本，允许为空
    cleaned['disease'] = str(data.get('disease', ''))

    # 家庭糖尿病史 (布尔值)
    try:
        val = data.get('family_diabetes', False)
        if isinstance(val, bool):
            cleaned['family_diabetes'] = val
        elif isinstance(val, str):
            cleaned['family_diabetes'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['family_diabetes'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "家族糖尿病史（family_diabetes）格式错误，请输入布尔值"

    # parental_osteoporosis (布尔值)
    try:
        val = data.get('parental_osteoporosis', False)
        if isinstance(val, bool):
            cleaned['parental_osteoporosis'] = val
        elif isinstance(val, str):
            cleaned['parental_osteoporosis'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['parental_osteoporosis'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "父母骨质疏松史（parental_osteoporosis）格式错误，请输入布尔值"

    # ==================== 体格测量 ====================
    try:
        cleaned['height_cm'] = float(data.get('height_cm', 165.0))
        if cleaned['height_cm'] < 50 or cleaned['height_cm'] > 250:
            return False, None, "身高（height_cm）超出合理范围（50-250cm）"
    except (TypeError, ValueError):
        return False, None, "身高（height_cm）格式错误，请输入数字"

    try:
        cleaned['weight_kg'] = float(data.get('weight_kg', 65.0))
        if cleaned['weight_kg'] < 10 or cleaned['weight_kg'] > 300:
            return False, None, "体重（weight_kg）超出合理范围（10-300kg）"
    except (TypeError, ValueError):
        return False, None, "体重（weight_kg）格式错误，请输入数字"

    try:
        cleaned['bmi'] = float(data.get('bmi', 23.9))
        if cleaned['bmi'] < 10 or cleaned['bmi'] > 80:
            return False, None, "身体质量指数（bmi）超出合理范围（10-80）"
    except (TypeError, ValueError):
        return False, None, "身体质量指数（bmi）格式错误，请输入数字"

    try:
        cleaned['body_fat_pct'] = float(data.get('body_fat_pct', 25.0))
        if cleaned['body_fat_pct'] < 3 or cleaned['body_fat_pct'] > 70:
            return False, None, "体脂率（body_fat_pct）超出合理范围（3-70%）"
    except (TypeError, ValueError):
        return False, None, "体脂率（body_fat_pct）格式错误，请输入数字"

    try:
        cleaned['bone_density'] = float(data.get('bone_density', 1.0))
        if cleaned['bone_density'] < 0.3 or cleaned['bone_density'] > 2.5:
            return False, None, "骨密度（bone_density）超出合理范围（0.3-2.5 g/cm²）"
    except (TypeError, ValueError):
        return False, None, "骨密度（bone_density）格式错误，请输入数字"

    # ==================== 生命体征 ====================
    try:
        cleaned['systolic_bp'] = float(data.get('systolic_bp', 120))
        if cleaned['systolic_bp'] < 60 or cleaned['systolic_bp'] > 260:
            return False, None, "收缩压（systolic_bp）超出合理范围（60-260 mmHg）"
    except (TypeError, ValueError):
        return False, None, "收缩压（systolic_bp）格式错误，请输入数字"

    try:
        cleaned['diastolic_bp'] = float(data.get('diastolic_bp', 80))
        if cleaned['diastolic_bp'] < 30 or cleaned['diastolic_bp'] > 160:
            return False, None, "舒张压（diastolic_bp）超出合理范围（30-160 mmHg）"
    except (TypeError, ValueError):
        return False, None, "舒张压（diastolic_bp）格式错误，请输入数字"

    try:
        cleaned['heart_rate'] = float(data.get('heart_rate', 75))
        if cleaned['heart_rate'] < 30 or cleaned['heart_rate'] > 200:
            return False, None, "脉搏（heart_rate）超出合理范围（30-200 次/分钟）"
    except (TypeError, ValueError):
        return False, None, "脉搏（heart_rate）格式错误，请输入数字"

    # heart_rhythm_regular (布尔值)
    try:
        val = data.get('heart_rhythm_regular', True)
        if isinstance(val, bool):
            cleaned['heart_rhythm_regular'] = val
        elif isinstance(val, str):
            cleaned['heart_rhythm_regular'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['heart_rhythm_regular'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "脉搏是否规律（heart_rhythm_regular）格式错误，请输入布尔值"

    # ==================== 血液常规检查 ====================
    try:
        cleaned['wbc'] = float(data.get('wbc', 6.0))
        if cleaned['wbc'] < 0.5 or cleaned['wbc'] > 50:
            return False, None, "白细胞计数（wbc）超出合理范围（0.5-50 10³/µL）"
    except (TypeError, ValueError):
        return False, None, "白细胞计数（wbc）格式错误，请输入数字"

    try:
        cleaned['rbc'] = float(data.get('rbc', 4.8))
        if cleaned['rbc'] < 1.0 or cleaned['rbc'] > 8.0:
            return False, None, "红细胞计数（rbc）超出合理范围（1.0-8.0 10⁶/µL）"
    except (TypeError, ValueError):
        return False, None, "红细胞计数（rbc）格式错误，请输入数字"

    try:
        cleaned['hemoglobin'] = float(data.get('hemoglobin', 14.0))
        if cleaned['hemoglobin'] < 4 or cleaned['hemoglobin'] > 25:
            return False, None, "血红蛋白（hemoglobin）超出合理范围（4-25 g/dL）"
    except (TypeError, ValueError):
        return False, None, "血红蛋白（hemoglobin）格式错误，请输入数字"

    try:
        cleaned['platelet'] = float(data.get('platelet', 250))
        if cleaned['platelet'] < 10 or cleaned['platelet'] > 1000:
            return False, None, "血小板计数（platelet）超出合理范围（10-1000 10³/µL）"
    except (TypeError, ValueError):
        return False, None, "血小板计数（platelet）格式错误，请输入数字"

    try:
        cleaned['total_cholesterol'] = float(data.get('total_cholesterol', 4.5))
        if cleaned['total_cholesterol'] < 1.0 or cleaned['total_cholesterol'] > 15.0:
            return False, None, "总胆固醇（total_cholesterol）超出合理范围（1.0-15.0 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "总胆固醇（total_cholesterol）格式错误，请输入数字"

    # ==================== 血液生化检查 ====================
    try:
        cleaned['glucose'] = float(data.get('glucose', 5.0))
        if cleaned['glucose'] < 1.0 or cleaned['glucose'] > 30.0:
            return False, None, "血糖（glucose）超出合理范围（1.0-30.0 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "血糖（glucose）格式错误，请输入数字"

    try:
        cleaned['triglycerides'] = float(data.get('triglycerides', 1.5))
        if cleaned['triglycerides'] < 0.1 or cleaned['triglycerides'] > 20.0:
            return False, None, "甘油三酯（triglycerides）超出合理范围（0.1-20.0 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "甘油三酯（triglycerides）格式错误，请输入数字"

    try:
        cleaned['creatinine'] = float(data.get('creatinine', 80))
        if cleaned['creatinine'] < 10 or cleaned['creatinine'] > 1500:
            return False, None, "肌酐（creatinine）超出合理范围（10-1500 µmol/L）"
    except (TypeError, ValueError):
        return False, None, "肌酐（creatinine）格式错误，请输入数字"

    try:
        cleaned['phosphorus'] = float(data.get('phosphorus', 1.2))
        if cleaned['phosphorus'] < 0.3 or cleaned['phosphorus'] > 3.0:
            return False, None, "磷（phosphorus）超出合理范围（0.3-3.0 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "磷（phosphorus）格式错误，请输入数字"

    try:
        cleaned['potassium'] = float(data.get('potassium', 4.0))
        if cleaned['potassium'] < 1.5 or cleaned['potassium'] > 8.0:
            return False, None, "钾（potassium）超出合理范围（1.5-8.0 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "钾（potassium）格式错误，请输入数字"

    try:
        cleaned['albumin'] = float(data.get('albumin', 42))
        if cleaned['albumin'] < 10 or cleaned['albumin'] > 70:
            return False, None, "白蛋白（albumin）超出合理范围（10-70 g/L）"
    except (TypeError, ValueError):
        return False, None, "白蛋白（albumin）格式错误，请输入数字"

    try:
        cleaned['bun'] = float(data.get('bun', 5.5))
        if cleaned['bun'] < 0.5 or cleaned['bun'] > 40:
            return False, None, "血尿素氮（bun）超出合理范围（0.5-40 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "血尿素氮（bun）格式错误，请输入数字"

    try:
        cleaned['bicarbonate'] = float(data.get('bicarbonate', 26))
        if cleaned['bicarbonate'] < 5 or cleaned['bicarbonate'] > 50:
            return False, None, "碳酸氢盐（bicarbonate）超出合理范围（5-50 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "碳酸氢盐（bicarbonate）格式错误，请输入数字"

    try:
        cleaned['sodium'] = float(data.get('sodium', 140))
        if cleaned['sodium'] < 100 or cleaned['sodium'] > 180:
            return False, None, "钠（sodium）超出合理范围（100-180 mmol/L）"
    except (TypeError, ValueError):
        return False, None, "钠（sodium）格式错误，请输入数字"

    try:
        cleaned['globulin'] = float(data.get('globulin', 30))
        if cleaned['globulin'] < 5 or cleaned['globulin'] > 60:
            return False, None, "球蛋白（globulin）超出合理范围（5-60 g/L）"
    except (TypeError, ValueError):
        return False, None, "球蛋白（globulin）格式错误，请输入数字"

    try:
        cleaned['insulin'] = float(data.get('insulin', 50))
        if cleaned['insulin'] < 0 or cleaned['insulin'] > 1000:
            return False, None, "胰岛素（insulin）超出合理范围（0-1000 pmol/L）"
    except (TypeError, ValueError):
        return False, None, "胰岛素（insulin）格式错误，请输入数字"

    # ==================== 尿液微量元素（µg/L） ====================
    try:
        cleaned['urine_barium'] = float(data.get('urine_barium', 1.0))
        if cleaned['urine_barium'] < 0 or cleaned['urine_barium'] > 1000:
            return False, None, "尿钡（urine_barium）超出合理范围（0-1000 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿钡（urine_barium）格式错误，请输入数字"

    try:
        cleaned['urine_cadmium'] = float(data.get('urine_cadmium', 0.1))
        if cleaned['urine_cadmium'] < 0 or cleaned['urine_cadmium'] > 100:
            return False, None, "尿镉（urine_cadmium）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿镉（urine_cadmium）格式错误，请输入数字"

    try:
        cleaned['urine_cobalt'] = float(data.get('urine_cobalt', 0.5))
        if cleaned['urine_cobalt'] < 0 or cleaned['urine_cobalt'] > 100:
            return False, None, "尿钴（urine_cobalt）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿钴（urine_cobalt）格式错误，请输入数字"

    try:
        cleaned['urine_cesium'] = float(data.get('urine_cesium', 5.0))
        if cleaned['urine_cesium'] < 0 or cleaned['urine_cesium'] > 500:
            return False, None, "尿铯（urine_cesium）超出合理范围（0-500 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿铯（urine_cesium）格式错误，请输入数字"

    try:
        cleaned['urine_manganese'] = float(data.get('urine_manganese', 0.5))
        if cleaned['urine_manganese'] < 0 or cleaned['urine_manganese'] > 100:
            return False, None, "尿锰（urine_manganese）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿锰（urine_manganese）格式错误，请输入数字"

    try:
        cleaned['urine_molybdenum'] = float(data.get('urine_molybdenum', 50.0))
        if cleaned['urine_molybdenum'] < 0 or cleaned['urine_molybdenum'] > 1000:
            return False, None, "尿钼（urine_molybdenum）超出合理范围（0-1000 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿钼（urine_molybdenum）格式错误，请输入数字"

    try:
        cleaned['urine_lead'] = float(data.get('urine_lead', 0.5))
        if cleaned['urine_lead'] < 0 or cleaned['urine_lead'] > 100:
            return False, None, "尿铅（urine_lead）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿铅（urine_lead）格式错误，请输入数字"

    try:
        cleaned['urine_antimony'] = float(data.get('urine_antimony', 0.1))
        if cleaned['urine_antimony'] < 0 or cleaned['urine_antimony'] > 100:
            return False, None, "尿锑（urine_antimony）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿锑（urine_antimony）格式错误，请输入数字"

    try:
        cleaned['urine_tin'] = float(data.get('urine_tin', 0.5))
        if cleaned['urine_tin'] < 0 or cleaned['urine_tin'] > 100:
            return False, None, "尿锡（urine_tin）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿锡（urine_tin）格式错误，请输入数字"

    try:
        cleaned['urine_thallium'] = float(data.get('urine_thallium', 0.2))
        if cleaned['urine_thallium'] < 0 or cleaned['urine_thallium'] > 100:
            return False, None, "尿铊（urine_thallium）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿铊（urine_thallium）格式错误，请输入数字"

    try:
        cleaned['urine_tungsten'] = float(data.get('urine_tungsten', 0.1))
        if cleaned['urine_tungsten'] < 0 or cleaned['urine_tungsten'] > 100:
            return False, None, "尿钨（urine_tungsten）超出合理范围（0-100 µg/L）"
    except (TypeError, ValueError):
        return False, None, "尿钨（urine_tungsten）格式错误，请输入数字"

    # ==================== 胸部症状 ====================
    # chest_pain (布尔值)
    try:
        val = data.get('chest_pain', False)
        if isinstance(val, bool):
            cleaned['chest_pain'] = val
        elif isinstance(val, str):
            cleaned['chest_pain'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['chest_pain'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "是否有胸痛（chest_pain）格式错误，请输入布尔值"

    # chest_pain_exercise (布尔值)
    try:
        val = data.get('chest_pain_exercise', False)
        if isinstance(val, bool):
            cleaned['chest_pain_exercise'] = val
        elif isinstance(val, str):
            cleaned['chest_pain_exercise'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['chest_pain_exercise'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "运动时胸痛（chest_pain_exercise）格式错误，请输入布尔值"

    # chest_pain_relief_10min (布尔值)
    try:
        val = data.get('chest_pain_relief_10min', False)
        if isinstance(val, bool):
            cleaned['chest_pain_relief_10min'] = val
        elif isinstance(val, str):
            cleaned['chest_pain_relief_10min'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['chest_pain_relief_10min'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "胸痛10分钟缓解（chest_pain_relief_10min）格式错误，请输入布尔值"

    # shortness_breath_exercise (布尔值)
    try:
        val = data.get('shortness_breath_exercise', False)
        if isinstance(val, bool):
            cleaned['shortness_breath_exercise'] = val
        elif isinstance(val, str):
            cleaned['shortness_breath_exercise'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['shortness_breath_exercise'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "运动时气短（shortness_breath_exercise）格式错误，请输入布尔值"

    # ==================== 饮食营养 ====================
    try:
        cleaned['daily_calories'] = float(data.get('daily_calories', 2000))
        if cleaned['daily_calories'] < 0 or cleaned['daily_calories'] > 20000:
            return False, None, "每日热量（daily_calories）超出合理范围（0-20000 kcal）"
    except (TypeError, ValueError):
        return False, None, "每日热量（daily_calories）格式错误，请输入数字"

    try:
        cleaned['daily_protein'] = float(data.get('daily_protein', 70))
        if cleaned['daily_protein'] < 0 or cleaned['daily_protein'] > 1000:
            return False, None, "每日蛋白质（daily_protein）超出合理范围（0-1000 g）"
    except (TypeError, ValueError):
        return False, None, "每日蛋白质（daily_protein）格式错误，请输入数字"

    try:
        cleaned['daily_carbs'] = float(data.get('daily_carbs', 250))
        if cleaned['daily_carbs'] < 0 or cleaned['daily_carbs'] > 2000:
            return False, None, "每日碳水（daily_carbs）超出合理范围（0-2000 g）"
    except (TypeError, ValueError):
        return False, None, "每日碳水（daily_carbs）格式错误，请输入数字"

    try:
        cleaned['daily_fiber'] = float(data.get('daily_fiber', 15))
        if cleaned['daily_fiber'] < 0 or cleaned['daily_fiber'] > 200:
            return False, None, "每日膳食纤维（daily_fiber）超出合理范围（0-200 g）"
    except (TypeError, ValueError):
        return False, None, "每日膳食纤维（daily_fiber）格式错误，请输入数字"

    try:
        cleaned['daily_fat'] = float(data.get('daily_fat', 70))
        if cleaned['daily_fat'] < 0 or cleaned['daily_fat'] > 500:
            return False, None, "每日脂肪（daily_fat）超出合理范围（0-500 g）"
    except (TypeError, ValueError):
        return False, None, "每日脂肪（daily_fat）格式错误，请输入数字"

    try:
        cleaned['daily_water'] = float(data.get('daily_water', 2000))
        if cleaned['daily_water'] < 0 or cleaned['daily_water'] > 20000:
            return False, None, "每日水摄入（daily_water）超出合理范围（0-20000 g）"
    except (TypeError, ValueError):
        return False, None, "每日水摄入（daily_water）格式错误，请输入数字"

    try:
        cleaned['salt_intake_level'] = int(data.get('salt_intake_level', 3))
        if cleaned['salt_intake_level'] not in range(1, 6):
            return False, None, "食盐摄入程度（salt_intake_level）请输入1-5之间的整数"
    except (TypeError, ValueError):
        return False, None, "食盐摄入程度（salt_intake_level）格式错误，请输入整数（1-5）"

    # diet_adequacy (布尔值)
    try:
        val = data.get('diet_adequacy', True)
        if isinstance(val, bool):
            cleaned['diet_adequacy'] = val
        elif isinstance(val, str):
            cleaned['diet_adequacy'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['diet_adequacy'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "饮食是否充足（diet_adequacy）格式错误，请输入布尔值"

    # diet_balance (布尔值)
    try:
        val = data.get('diet_balance', True)
        if isinstance(val, bool):
            cleaned['diet_balance'] = val
        elif isinstance(val, str):
            cleaned['diet_balance'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['diet_balance'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "饮食是否均衡（diet_balance）格式错误，请输入布尔值"

    # skip_meals (布尔值)
    try:
        val = data.get('skip_meals', False)
        if isinstance(val, bool):
            cleaned['skip_meals'] = val
        elif isinstance(val, str):
            cleaned['skip_meals'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['skip_meals'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "是否跳过正餐（skip_meals）格式错误，请输入布尔值"

    # ==================== 饮酒与吸烟 ====================
    try:
        cleaned['alcohol_frequency'] = int(data.get('alcohol_frequency', 1))
        if cleaned['alcohol_frequency'] not in range(1, 5):
            return False, None, "饮酒频率（alcohol_frequency）请输入1-4之间的整数"
    except (TypeError, ValueError):
        return False, None, "饮酒频率（alcohol_frequency）格式错误，请输入整数（1-4）"

    try:
        cleaned['smoking_frequency'] = int(data.get('smoking_frequency', 1))
        if cleaned['smoking_frequency'] not in range(1, 5):
            return False, None, "吸烟频率（smoking_frequency）请输入1-4之间的整数"
    except (TypeError, ValueError):
        return False, None, "吸烟频率（smoking_frequency）格式错误，请输入整数（1-4）"

    try:
        cleaned['smoking_start_age'] = int(data.get('smoking_start_age', 18))
        if cleaned['smoking_start_age'] < 0 or cleaned['smoking_start_age'] > 80:
            return False, None, "开始吸烟年龄（smoking_start_age）超出合理范围（0-80岁）"
    except (TypeError, ValueError):
        return False, None, "开始吸烟年龄（smoking_start_age）格式错误，请输入整数"

    # smoker_in_household (布尔值)
    try:
        val = data.get('smoker_in_household', False)
        if isinstance(val, bool):
            cleaned['smoker_in_household'] = val
        elif isinstance(val, str):
            cleaned['smoker_in_household'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['smoker_in_household'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "家中有人吸烟（smoker_in_household）格式错误，请输入布尔值"

    # secondhand_smoke (布尔值)
    try:
        val = data.get('secondhand_smoke', False)
        if isinstance(val, bool):
            cleaned['secondhand_smoke'] = val
        elif isinstance(val, str):
            cleaned['secondhand_smoke'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['secondhand_smoke'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "吸入二手烟（secondhand_smoke）格式错误，请输入布尔值"

    # ==================== 体力活动 ====================
    # physical_labor (布尔值)
    try:
        val = data.get('physical_labor', False)
        if isinstance(val, bool):
            cleaned['physical_labor'] = val
        elif isinstance(val, str):
            cleaned['physical_labor'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['physical_labor'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "从事体力劳动（physical_labor）格式错误，请输入布尔值"

    # exercise_weekly (布尔值)
    try:
        val = data.get('exercise_weekly', True)
        if isinstance(val, bool):
            cleaned['exercise_weekly'] = val
        elif isinstance(val, str):
            cleaned['exercise_weekly'] = val.lower() in ('true', '1', 'yes', 'y')
        else:
            cleaned['exercise_weekly'] = bool(val)
    except (TypeError, ValueError):
        return False, None, "每周锻炼（exercise_weekly）格式错误，请输入布尔值"

    try:
        cleaned['exercise_duration_min'] = float(data.get('exercise_duration_min', 30))
        if cleaned['exercise_duration_min'] < 0 or cleaned['exercise_duration_min'] > 1440:
            return False, None, "每天锻炼时长（exercise_duration_min）超出合理范围（0-1440分钟）"
    except (TypeError, ValueError):
        return False, None, "每天锻炼时长（exercise_duration_min）格式错误，请输入数字"

    try:
        cleaned['sedentary_duration_min'] = float(data.get('sedentary_duration_min', 480))
        if cleaned['sedentary_duration_min'] < 0 or cleaned['sedentary_duration_min'] > 1440:
            return False, None, "每天久坐时长（sedentary_duration_min）超出合理范围（0-1440分钟）"
    except (TypeError, ValueError):
        return False, None, "每天久坐时长（sedentary_duration_min）格式错误，请输入数字"

    # ==================== 睡眠 ====================
    try:
        cleaned['sleep_hours_workday'] = float(data.get('sleep_hours_workday', 7.5))
        if cleaned['sleep_hours_workday'] < 0 or cleaned['sleep_hours_workday'] > 24:
            return False, None, "工作日睡眠（sleep_hours_workday）超出合理范围（0-24小时）"
    except (TypeError, ValueError):
        return False, None, "工作日睡眠（sleep_hours_workday）格式错误，请输入数字"

    try:
        cleaned['snoring_frequency'] = int(data.get('snoring_frequency', 1))
        if cleaned['snoring_frequency'] not in range(1, 4):
            return False, None, "打鼾频率（snoring_frequency）请输入1-3之间的整数"
    except (TypeError, ValueError):
        return False, None, "打鼾频率（snoring_frequency）格式错误，请输入整数（1-3）"

    # ==================== 其他 ====================
    try:
        cleaned['night_urination'] = int(data.get('night_urination', 1))
        if cleaned['night_urination'] < 0 or cleaned['night_urination'] > 20:
            return False, None, "起夜次数（night_urination）超出合理范围（0-20次）"
    except (TypeError, ValueError):
        return False, None, "起夜次数（night_urination）格式错误，请输入整数"

    # 校验通过，返回清洗后的数据
    return True, cleaned, None
'''
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
        """
        转换并校验前端数据
        根据 config 校验并填充缺失值
        """
        type_map = {
            "int": int,
            "float": float,
            "bool": lambda x: bool(x) if isinstance(x,bool) else str(x).lower() in ('true', '1', 'yes'),
            "str": str
        }

        result = {}
        for key, field_config in FEATURE_COLS_and_Default_VALUES.items():
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

        return result


def build_feature_vector(data:dict)->list:
    features = []
    for col, field_config in FEATURE_COLS_and_Default_VALUES.items():  # ← 使用config中定义的顺序
        if col == "BMI（kg/m^2）":
            height = data.get('身高（cm）')
            weight = data.get('体重（kg）')
            value = weight / ((height / 100) ** 2)
        else:
            value = data.get(col, field_config.get('default', 0))
        features.append(float(value))
    return [features]
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
def do_prediction(data:list)->list:
    '''
    disease = function(data)
    return disease
    '''