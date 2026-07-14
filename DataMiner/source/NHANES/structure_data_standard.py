import pandas

from get_data import DATA_PATH , DATA_NAMES , SHEET_MAP

DATA_MAP = {
    'SEQN':{'type': 'int'},
    '患病情况':{'type': 'str'},
    '饮酒频率':{'type': 'int'},
    '血压收缩压（mmHg）':{'type': 'float'},
    '血压舒张压（mmHg）':{'type': 'float'},
    '脉搏（次/分钟）':{'type': 'int'},
    '脉搏是否规律':{'type': 'bool'},
    '体重（kg）':{'type': 'float'},
    '身高（cm）':{'type': 'float'},
    'BMI（kg/m²）':{'type': 'float'},
    '是否有过胸部疼痛':{'type': 'bool'},
    '是否在运动时胸痛':{'type': 'bool'},
    '疼痛能否在10分钟内缓解':{'type': 'bool'},
    '运动时是否气短':{'type': 'bool'},
    '胆固醇（mg/dL）':{'type': 'int'},
    '白细胞计数（1000个细胞/uL）':{'type': 'float'},
    '红细胞计数（10⁶ 个细胞/µL）':{'type': 'float'},
    '血红蛋白（g/dL）':{'type': 'float'},
    '血小板计数（1000 个细胞/µL）':{'type': 'int'},
    '性别':{'type': 'str'},
    '存在家族糖尿病史':{'type': 'bool'},
    '每日食盐摄入程度':{'type': 'int'},
    '每日热量摄入量（kcal）':{'type': 'int'},
    '每日蛋白质摄入量（g）':{'type': 'float'},
    '每日碳水摄入量（g）':{'type': 'float'},
    '每日膳食纤维摄入量（g）':{'type': 'float'},
    '每日脂肪摄入量（g）':{'type': 'float'},
    '每日水摄入量（g）':{'type': 'int'},
    '骨密度（g/cm²）':{'type': 'float'},
    '体脂率（%）':{'type': 'float'},
    '饮食是否充足':{'type': 'int'},
    '饮食是否均衡':{'type': 'bool'},
    '是否会跳过正餐':{'type': 'bool'},
    '胰岛素（pmol/L）':{'type': 'float'},
    '起夜次数':{'type': 'int'},
    '尿检·钡（µg/L）':{'type': 'float'},
    '尿检·镉（µg/L）':{'type': 'float'},
    '尿检·钴（µg/L）':{'type': 'float'},
    '尿检·铯（µg/L）':{'type': 'float'},
    '尿检·锰（µg/L）':{'type': 'float'},
    '尿检·钼（µg/L）':{'type': 'float'},
    '尿检·铅（µg/L）':{'type': 'float'},
    '尿检·锑（µg/L）':{'type': 'float'},
    '尿检·锡（µg/L）':{'type': 'float'},
    '尿检·铊（µg/L）':{'type': 'float'},
    '尿检·钨（µg/L）':{'type': 'float'},
    '父母是否患有骨质疏松':{'type': 'bool'},
    '是否从事体力劳动':{'type': 'bool'},
    '是否每周进行进行中等以上强度锻炼':{'type': 'bool'},
    '平均每天中等以上强度锻炼时长（分钟）':{'type': 'int'},
    '平均每天坐姿时长（分钟）':{'type': 'int'},
    '工作日平均睡眠小时数':{'type': 'float'},
    '打鼾频率':{'type': 'int'},
    '吸烟频率':{'type': 'int'},
    '开始吸烟年龄':{'type': 'int'},
    '家中是否有人吸烟':{'type': 'bool'},
    '一周内是否吸入过二手烟':{'type': 'bool'},
    '葡萄糖（mg/dL）':{'type': 'int'},
    '甘油三酯（mg/dL）':{'type': 'int'},
    '肌酐（mg/dL）':{'type': 'float'},
    '磷（mg/dL）':{'type': 'int'},
    '钾（mmol/L）':{'type': 'float'},
    '白蛋白（g/dL）':{'type': 'float'},
    '血尿素氮（mg/dL）':{'type': 'int'},
    '碳酸氢盐（mmol/L）':{'type': 'int'},
    '钠（mmol/L）':{'type': 'int'},
    '球蛋白（g/dL）':{'type': 'float'},
}

data_file = pandas.read_csv( DATA_PATH[:-1] + '.csv')

# 读取表，列出字典格式。
def get_column_list():
    for column in data_file.columns:
        print("\t'" , column , "':{'type': ''}," , sep = '')

# 计算标准值
def calculate_standards():
    target_path = DATA_PATH[:-1] + '_standard.csv'
    for column in DATA_MAP:
        if column == 'SEQN':
            continue
        elif column == '患病情况':
            for line in data_file[column]:
                diseases = str.split(line , ';')
                if diseases is None or len(diseases) == 0:
                    continue
                for disease in diseases:

            pass
        else:
            type = DATA_MAP[column]['type']
            if type == 'bool':
                pass
            elif type == 'int':
                pass
            elif type == 'float':
                pass
            elif type == 'str':
                pass