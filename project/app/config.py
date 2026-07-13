# config.py
# 所有配置集中管理

# 特征顺序（和模型开发人员商定的）
FEATURE_COLS = [
    'age', 'gender', 'height', 'weight', 'bmi',
    'systolic_bp', 'diastolic_bp', 'heart_rate',
    'fasting_glucose', 'total_cholesterol', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides',
    'alt', 'ast', 'creatinine', 'urea'
]

# 默认值
DEFAULT_VALUES = {
    'height': 165,
    'weight': 65,
    'heart_rate': 75,
    'total_cholesterol': 4.5,
    'hdl_cholesterol': 1.3,
    'ldl_cholesterol': 2.8,
    'triglycerides': 1.5,
    'alt': 25,
    'ast': 22,
    'creatinine': 80,
    'urea': 5.5
}

# 服务配置
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True