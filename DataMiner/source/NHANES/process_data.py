import os.path

import pandas

from get_data import DATA_PATH , DATA_NAMES , SHEET_MAP

def salt_intake(DRQSPREP):
    if DRQSPREP is None or DRQSPREP == 9:
        return None
    return DRQSPREP



# 需要的数据项以及具体处理方法
# 每个数据项用元组表示，元组内元素分别为 (处理后名称，数据项代号/考虑多项时为数组，是否迭代处理/即是否需要将未处理值与处理后目标项共同处理，处理函数/为None时无需处理），说明信息
INTAKE_MAP = {
    'Alcohol Use':[

    ],
    'Audiometry':[],
    'Blood Pressure & Cholesterol':[],
    'Body Measures':[],
    'Cardiovascular Health':[],
    'Cholesterol - Total':[],
    'Complete Blood Count with 5-Part Differential':[],
    'Current Health Status':[],
    'Dermatology':[],
    'Diabetes':[],
    'Diet Behavior & Nutrition':[],
    'Dietary Interview - Total Nutrient Intakes, First Day':[
        ('食盐摄入程度' , ['DRQSPREP'] , False , salt_intake , '衡量食盐摄入量标准，值从 0 ~ 4 表示从不添加盐到加大量盐，None 表示缺失'),
        ('热量摄入量' , ['DR1TKCAL'] , 'kcal'),
        ('蛋白质摄入量' , ['DR1TPROT'] , 'g'),
        ('碳水摄入量' , ['DR1TCARB'] , 'g'),
        ('膳食纤维摄入量' , ['DR1TFIBE'] , 'g'),
        ('脂肪摄入量' , ['DR1TTFAT'] , 'g'),
        ('水摄入量' , ['DR1_320Z'] , 'g'),
    ],
    'Food Security':[],
    'Glycohemoglobin':[],
    'Hepatitis':[],
    'High-Sensitivity C-Reactive Protein':[],
    'Housing Characteristics':[],
    'Insulin':[],
    'Iodine - Urine':[],
    'Kidney Conditions - Urology':[],
    'Medical Conditions':[],
    'Metals - Urine':[],
    'Oral Health':[],
    'Osteoporosis':[],
    'Physical Activity - Youth':[],
    'Physical Activity':[],
    'Sex Steroid Hormone Panel - Serum (Surplus)':[],
    'Sleep Disorders':[],
    'Smoking - Cigarette Use':[],
    'Smoking - Household Smokers':[],
    'Smoking - Recent Tobacco Use':[],
    'Smoking - Secondhand Smoke Exposure':[],
    'Standard Biochemistry Profile':[],
    'Weight History - Youth':[],
    'Weight History':[],
}


def process() -> pandas.DataFrame:
    # 初始化结果集
    data_file = pandas.read_csv(DATA_PATH + "Dietary Interview - Total Nutrient Intakes, First Day.csv" , encoding = 'utf-8' , index_col='SEQN')
    result = pandas.DataFrame(index = data_file.index)
    for name , list in INTAKE_MAP.items():
        path = DATA_PATH + name + ".csv"
        if not os.path.exists(path):
            print('> Warning: File "' , name , '.csv does not exist.' , sep = '')
            continue
        data_file = pandas.read_csv(path , encoding = 'utf-8' , index_col = 'SEQN')
        for process in list:
            print('> Processing column "' , process[0] , '"' , sep = '')
            if len(process) < 4:
                # 参数缺失，单数据转移
                result[[process[0]]] = data_file[[process[1]]]
                continue
            if type(process[1]) is str:
                # 处理单数据项
                if process[2] is True:
                    # 处理单数据迭代
                    result[process[0]] = process[3](result[process[0]] , data_file[process[1]])
                else:
                    # 处理独立单数据项
                    if process[3] is None:
                        # 处理单数据转移
                        result[[process[0]]] = data_file[[process[1]]]
                    else:
                        # 处理独立单数据
                        result[[process[0]]] = process[3](data_file[[process[1]]])
            else:
                # 处理多数据项
                column_list = []
                for column in process[1]:
                    column_list.append(data_file[column])
                if process[2] is True:
                    # 处理多数据迭代
                    result[process[0]] = process[3](result[process[0]] , *column_list)
                else:
                    # 处理多数据
                    result[process[0]] = process[3](*column_list)
    return result

result = process()
result.to_csv(DATA_PATH[:-1] + ".csv" , encoding = 'utf-8')