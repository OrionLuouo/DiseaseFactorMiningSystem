import json
import os

import math
import pandas
from pandas.core.interchange.dataframe_protocol import DataFrame

from get_data import DATA_PATH
from process_data import INTAKE_MAP

# 默认的缺失容许率
DEFAULT_MISSING_TOLERANCE = 0.5

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
    'BMI（kg/m²）':{'type': 'float'  , 'tolerance': 1.1 , 'necessary': True , 'function': (['体重（kg）' , '身高（cm）'] , lambda x , y: x / y / y * 10000)},
    '是否有过胸部疼痛':{'type': 'bool'},
    '是否在运动时胸痛':{'type': 'bool' , 'tolerance': -1},
    '疼痛能否在10分钟内缓解':{'type': 'bool' , 'tolerance': -1},
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
    '骨密度（g/cm²）':{'type': 'float' , 'tolerance': -1},
    '体脂率（%）':{'type': 'float' , 'tolerance': -1},
    '饮食是否充足':{'type': 'int'},
    '饮食是否均衡':{'type': 'bool'},
    '是否会跳过正餐':{'type': 'bool' , 'tolerance': -1},
    '胰岛素（pmol/L）':{'type': 'float' , 'tolerance': -1},
    '起夜次数':{'type': 'int'},
    '尿检·钡（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·镉（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·钴（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·铯（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·锰（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·钼（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·铅（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·锑（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·锡（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·铊（µg/L）':{'type': 'float' , 'tolerance': -1},
    '尿检·钨（µg/L）':{'type': 'float' , 'tolerance': -1},
    '父母是否患有骨质疏松':{'type': 'bool' , 'tolerance': -1},
    '是否从事体力劳动':{'type': 'bool'},
    '是否每周进行进行中等以上强度锻炼':{'type': 'bool'},
    '平均每天中等以上强度锻炼时长（分钟）':{'type': 'int' , 'tolerance': -1},
    '平均每天坐姿时长（分钟）':{'type': 'int'},
    '工作日平均睡眠小时数':{'type': 'float'},
    '打鼾频率':{'type': 'int'},
    '吸烟频率':{'type': 'int'},
    '开始吸烟年龄':{'type': 'int' , 'tolerance': -1},
    '家中是否有人吸烟':{'type': 'bool' , 'tolerance': -1},
    '一周内是否吸入过二手烟':{'type': 'bool' , 'tolerance': -1},
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

CATEGORY_MAP = [
    {
        'category': '基本信息',
        'data_list': [
            {
                'data': '性别'
            },
            {
                'data': '体重（kg）'
            },
            {
                'data': '身高（cm）'
            },
            {
                'data': 'BMI（kg/m²）'
            },
            {
                'data': '是否从事体力劳动'
            }
        ]
    },
    {
        'category': '疾病史',
        'data_list': [
            {
                'data': '患病情况'
            } ,
            {
                'data': '存在家族糖尿病史'
            }
        ]
    },
    {
        'category': '生活规律',
        'data_list': [
            {
                'data': '饮酒频率'
            },
            {
                'data': '饮食是否充足'
            },
            {
                'data': '饮食是否均衡'
            },
            {
                'data': '起夜次数'
            },
            {
                'data': '是否每周进行进行中等以上强度锻炼'
            },
            {
                'data': '平均每天坐姿时长（分钟）'
            },
            {
                'data': '工作日平均睡眠小时数'
            },
            {
                'data': '打鼾频率'
            }
        ]
    },
    {
        'category': '饮食习惯',
        'data_list': [
            {
                'data': '每日食盐摄入程度'
            },
            {
                'data': '每日热量摄入量（kcal）'
            },
            {
                'data': '每日蛋白质摄入量（g）'
            },
            {
                'data': '每日碳水摄入量（g）'
            },
            {
                'data': '每日膳食纤维摄入量（g）'
            },
            {
                'data': '每日脂肪摄入量（g）'
            },
            {
                'data': '每日水摄入量（g）'
            },
        ]
    },
    {
        'category': '体检数据',
        'data_list': [
            {
                'data': '血压收缩压（mmHg）'
            },
            {
                'data': '血压舒张压（mmHg）'
            },
            {
                'data': '脉搏（次/分钟）'
            },
            {
                'data': '脉搏是否规律'
            }
        ]
    },
    {
        'category': '化验数据',
        'data_list': [
            {
                'data': '胆固醇（mg/dL）'
            },
            {
                'data': '葡萄糖（mg/dL）'
            },
            {
                'data': '甘油三酯（mg/dL）'
            },
            {
                'data': '肌酐（mg/dL）'
            },
            {
                'data': '磷（mg/dL）'
            },
            {
                'data': '钾（mmol/L）'
            },
            {
                'data': '白蛋白（g/dL）'
            },
            {
                'data': '血尿素氮（mg/dL）'
            },
            {
                'data': '碳酸氢盐（mmol/L）'
            },
            {
                'data': '钠（mmol/L）'
            },
            {
                'data': '球蛋白（g/dL）'
            },
        ]
    },
    {
        'category': '血检数据',
        'data_list': [
            {
                'data': '白细胞计数（1000个细胞/uL）'
            },
            {
                'data': '红细胞计数（10⁶ 个细胞/µL）'
            },
            {
                'data': '血红蛋白（g/dL）'
            },
            {
                'data': '血小板计数（1000 个细胞/µL）'
            },
        ]
    }
]

DROP_LIST = ['SEQN']

data_file: DataFrame = None

# 读取表，列出字典格式。
def get_column_list():
    for column in data_file.columns:
        print("\t'" , column , "':{'type': ''}," , sep = '')

# 按照缺失处理规则清洗数据
def clean_data():
    drop_list = []
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
            drop_list.append(column)
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
    DATA_MAP.pop('SEQN')
    for drop_column in drop_list:
        DATA_MAP.pop(drop_column)
    for column in DROP_LIST:
        if column in data_file.columns:
            data_file.drop(column, axis=1, inplace=True)

# 计算标准
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
    print('> 数据标准已写入文件 "' , str.split(target_path , '/')[-1] , sep = '')
    for column in data_file.columns:
        data_file[column] = data_file[column].fillna(DATA_MAP[column]['default'])
    float_columns = data_file.select_dtypes(include=['float']).columns
    for column in float_columns:
        data_file[column] = data_file[column].round(4)
    target_path = DATA_PATH[:-1] + '_cleaned.csv'
    if os.path.exists(target_path):
        os.remove(target_path)
    data_file.to_csv(target_path , index=False , encoding='utf-8')
    print('> 数据清洗结果已写入文件 "' , str.split(target_path , '/')[-1] , sep = '')

# 格式化分类表
def categorize():
    information_map = {}
    for data_list in INTAKE_MAP:
        for data_tuple in INTAKE_MAP[data_list]:
            if len(data_tuple) == 5:
                information_map[data_tuple[0]] = data_tuple[4]
    for category in CATEGORY_MAP:
        data_list = category['data_list']
        for data in data_list:
            name = data['data']
            data['type'] = DATA_MAP[name]['type']
            data['default'] = DATA_MAP[name]['default']
            if 'necessary' in DATA_MAP[name]:
                data['necessary'] = DATA_MAP[name]['necessary']
            if name in information_map:
                data['information'] = information_map[name]
    target_path = DATA_PATH[:-1] + '_category.json'
    with open(target_path , 'w' , encoding = 'utf-8') as target_file:
        json.dump(CATEGORY_MAP , target_file , ensure_ascii = False , indent = 4)
    print('> 数据分类标准已写入文件 "' , str.split(target_path , '/')[-1] , sep = '')

def run():
    print('> Standardizing data...')
    global data_file
    data_file = pandas.read_csv(DATA_PATH[:-1] + '.csv', encoding='utf-8', index_col='SEQN')
    clean_data()
    calculate_standards()
    categorize()

if __name__ == '__main__':
    run()