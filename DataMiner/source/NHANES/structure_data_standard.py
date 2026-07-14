import json
import os

import math
import pandas

from get_data import DATA_PATH , DATA_NAMES , SHEET_MAP
from source.NHANES.process_data import TARGET_PATH

# 默认的缺失容许率
DEFAULT_MISSING_TOLERANCE = 0.4

DATA_MAP = {
    'SEQN':{'type': 'int' , 'tolerance': 1.1},
    '患病情况':{'type': 'str' , 'tolerance': 1.1},
    '饮酒频率':{'type': 'int'},
    '血压收缩压（mmHg）':{'type': 'float'},
    '血压舒张压（mmHg）':{'type': 'float'},
    '脉搏（次/分钟）':{'type': 'int'},
    '脉搏是否规律':{'type': 'bool'},
    '体重（kg）':{'type': 'float' , 'necessary': True},
    '身高（cm）':{'type': 'float' , 'necessary': True},
    'BMI（kg/m²）':{'type': 'float' , 'necessary': True , 'function': (['体重（kg）' , '身高（cm）'] , lambda x , y: x / y / y * 10000)},
    '是否有过胸部疼痛':{'type': 'bool'},
    '是否在运动时胸痛':{'type': 'bool'},
    '疼痛能否在10分钟内缓解':{'type': 'bool'},
    '运动时是否气短':{'type': 'bool'},
    '胆固醇（mg/dL）':{'type': 'int'},
    '白细胞计数（1000个细胞/uL）':{'type': 'float'},
    '红细胞计数（10⁶ 个细胞/µL）':{'type': 'float'},
    '血红蛋白（g/dL）':{'type': 'float'},
    '血小板计数（1000 个细胞/µL）':{'type': 'int'},
    '性别':{'type': 'str' , 'tolerance': 1.1 , 'necessary': True},
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

DROP_LIST = ['SEQN']

data_file = pandas.read_csv(DATA_PATH[:-1] + '.csv' , encoding = 'utf-8' , index_col = 'SEQN')

# 读取表，列出字典格式。
def get_column_list():
    for column in data_file.columns:
        print("\t'" , column , "':{'type': ''}," , sep = '')

# 按照缺失处理规则清洗数据
def clean_data():
    for column in DATA_MAP:
        if column not in data_file.columns:
            continue
        missing_ratio = data_file[column].isna().mean()
        missing_tolerance:float = DEFAULT_MISSING_TOLERANCE
        if 'tolerance' in DATA_MAP[column]:
            missing_tolerance = DATA_MAP[column]['tolerance']
        if missing_ratio > missing_tolerance:
            print('> 字段 "' , column , '" 缺失率 ' , missing_ratio , ' 超过缺失容许 ' , missing_tolerance , sep = '')
            data_file.drop(column , axis=1 , inplace=True)
            continue
        if 'necessary' in DATA_MAP[column] and DATA_MAP[column]['necessary'] == True:
            if 'function' in DATA_MAP[column]:
                function_tuple = DATA_MAP[column]['function']
                missing_mask = data_file[column].isna()
                if missing_mask.any():
                    data_file.loc[missing_mask , column] = data_file.loc[missing_mask , function_tuple[0]].apply(lambda row: function_tuple[1](*row), axis=1)
            else:
                data_file.dropna(subset = [column] , inplace = True)
    data_file.reset_index(drop = True , inplace = True)
    for column in DROP_LIST:
        if column in data_file.columns:
            data_file.drop(column, axis=1, inplace=True)
    target_path = TARGET_PATH[:-1] + '_cleaned.csv'
    if os.path.exists(target_path):
        os.remove(target_path)
    data_file.to_csv(target_path , index=False , encoding='utf-8')
    print('> 数据清洗结果已存入文件 "' , str.split(target_path , '/')[-1] , sep = '')

# 计算标准值
def calculate_standards():
    count = len(data_file)
    for column in DATA_MAP:
        if 'function' in DATA_MAP[column]:
            # 删除 function 字段
            DATA_MAP[column].pop('function')
        if 'tolerance' in DATA_MAP[column]:
            # 删除 tolerance 字段
            DATA_MAP[column].pop('tolerance')
        if column not in data_file.columns:
            continue
        if column == 'SEQN':
            continue
        elif column == '患病情况':
            disease_map = {}
            for line in data_file[column]:
                if line is None or not type(line) == str:
                    continue
                diseases = str.split(line , ';')
                for disease in diseases:
                    if disease not in disease_map:
                        disease_map[disease] = 1
                    else:
                        disease_map[disease] = disease_map[disease] + 1
            value = ''
            for disease , disease_count in disease_map.items():
                if len(value) != 0:
                    value += ';'
                value += disease
                value += '='
                value += str(disease_count / count)
            DATA_MAP[column]['default'] = ''
            DATA_MAP[column]['value'] = value
        else:
            data_type = DATA_MAP[column]['type']
            if data_type == 'bool':
                true_count = 0
                for line in data_file[column]:
                    if line == True:
                        true_count += 1
                DATA_MAP[column]['default'] = True if (true_count / count) > 0.5 else False
                DATA_MAP[column]['value'] = true_count / count
            elif data_type == 'int':
                accumulator = 0
                for line in data_file[column]:
                    if not math.isnan(line):
                        accumulator += line
                DATA_MAP[column]['default'] = round(accumulator / count)
                DATA_MAP[column]['value'] = accumulator / count
            elif data_type == 'float':
                accumulator = 0
                for line in data_file[column]:
                    if not math.isnan(line):
                        accumulator += line
                DATA_MAP[column]['default'] = accumulator / count
                DATA_MAP[column]['value'] = accumulator / count
            elif data_type == 'str':
                DATA_MAP[column]['default'] = ''
                continue
    target_path = DATA_PATH[:-1] + '_standard.json'
    with open(target_path , 'w' , encoding = 'utf-8') as target_file:
        json.dump(DATA_MAP , target_file , ensure_ascii = False , indent = 4)
    print('> 数据标准已写入 "' , str.split(target_path , '/')[-1] , sep = '')

clean_data()
calculate_standards()