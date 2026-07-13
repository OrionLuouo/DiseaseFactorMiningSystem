import os.path

import pandas

from get_data import DATA_PATH , DATA_NAMES , SHEET_MAP

# 布尔值检查，1 = True，2 = False，None = None
def bool_check(x:int):
    if x is None:
        return None
    if x > 2:
        return None
    return x == 1

# 检查患病信号，仅当为 1 时返回 True，其他情况返回 None
def disease_check(signal):
    return True if signal == 1 else None

# 添加已患疾病
def append_disease(disease:str , bool_function = disease_check, true_signal:int = 1):
    if disease is None or len(disease) == 0:
        return lambda x , y: x
    def append(current:str , signal:str):
        signal = bool_function(signal)
        if signal is None or signal == False:
            return current
        else:
            if current is None or not type(current) == str or len(current) == 0:
                return disease
            return current + ';' + disease
    return append
    
# 生成添加已患疾病的元组
def disease(code:str, disease_name:str, true_signal:int = 1) -> tuple:
    return DISEASE_COLUMN , code , True , append_disease(disease_name, true_signal = true_signal)

# 饮酒频率
def alcohol_frequency(x:int):
    if 1 <= x < 3:
        return 4
    elif x < 5:
        return 3
    elif x < 7:
        return 2
    elif x <= 10:
        return 1
    elif x == 0:
        return 0
    else:
        return None

def metal_urine(code:str , metal_name:str) -> tuple:
    return '尿检·' + metal_name + '（µg/L）' , code

# 患病情况字段的名称
DISEASE_COLUMN = '患病情况'

# 需要的数据项以及具体处理方法
# 每个数据项用元组表示，元组内元素分别为 (处理后名称，数据项代号/考虑多项时为数组，是否迭代处理/即是否需要将未处理值与处理后目标项共同处理，处理函数/为None时无需处理），说明信息
INTAKE_MAP = {
    'Alcohol Use':[
        ('饮酒频率' , 'ALQ121' , False , alcohol_frequency , '0 表示从不喝酒，1 表示一年数次，2 表示每月数次，3 表示每周数次，4 表示基本每日。')
    ],
    'Audiometry':[],
    'Blood Pressure':[
        ('血压收缩压（mmHg）' , ['BPXSY1' , 'BPXSY2' , 'BPXSY3'] , False , lambda x , y , z: (x + y + z) / 3),
        ('血压舒张压（mmHg）' , ['BPXDI1' , 'BPXDI2' , 'BPXDI3'] , False , lambda x , y , z: (x + y + z) / 3),
        ('脉搏（次/分钟）' , ['BPXCHR' , 'BPXPLS'] , False , lambda x , y: x if y is None else y),
        ('脉搏是否规律' , ['BPXPULS'] , False , bool_check)
    ],
    'Blood Pressure & Cholesterol':[],
    'Body Measures':[
        ('体重（kg）' , 'BMXWT'),
        ('身高（cm）' , 'BMXHT'),
        ('BMI（kg/m²）' , 'BMXBMI')
    ],
    'Cardiovascular Health':[
        # 心血管
        ('是否有过胸部疼痛' , 'CDQ001' , False , bool_check),
        ('是否在运动时胸痛' , ['CDQ002' , 'CDQ003'] , False , lambda x , y: True if x == 1 or y == 1 else None if x is None and y is None else False),
        ('疼痛能否在10分钟内缓解' , 'CDQ006' , False , bool_check),
        ('运动时是否气短' , 'CDQ010' , False , bool_check)
    ],
    'Cholesterol - Total':[
        ('胆固醇（mg/dL）' , 'LBXTC')
    ],
    'Complete Blood Count with 5-Part Differential':[
        ('白细胞计数（1000个细胞/uL）' , 'LBXWBCSI'),
        ('红细胞计数（10⁶ 个细胞/µL）' , 'LBXRBCSI'),
        ('血红蛋白（g/dL）' , 'LBXHGB'),
        ('血小板计数（1000 个细胞/µL）' , 'LBXPLTSI')
    ],
    'Current Health Status':[
        # 常见病
        (DISEASE_COLUMN , 'HSQ500' , True , append_disease('感冒' , disease_check)),
        (DISEASE_COLUMN , 'HSQ510' , True , append_disease('肠胃炎' , disease_check))
    ],
    'Dermatology':[],
    'Diabetes':[
        # 糖尿病
        (DISEASE_COLUMN , 'DIQ010' , True , append_disease('糖尿病' , lambda signal: True if signal == 1 else None)),
        ('存在家族糖尿病史' , 'DIQ170' , False , bool_check)
    ],
    'Diet Behavior & Nutrition':[],
    'Dietary Interview - Total Nutrient Intakes, First Day':[
        ('每日食盐摄入程度' , 'DRQSPREP' , False , lambda x: None if x == 9 else x , ' 从 1 ~ 4 表示程度轻重，1 表示从不添加盐，4 表示摄入很多盐。'),
        ('每日热量摄入量（kcal）' , 'DR1TKCAL' , ''),
        ('每日蛋白质摄入量（g）' , 'DR1TPROT' , ''),
        ('每日碳水摄入量（g）' , 'DR1TCARB' , ''),
        ('每日膳食纤维摄入量（g）' , 'DR1TFIBE' , ''),
        ('每日脂肪摄入量（g）' , 'DR1TTFAT' , ''),
        ('每日水摄入量（g）' , 'DR1_320Z' , ''),
    ],
    'Dual-Energy X-ray Absorptiometry - Whole Body':[
        ('骨密度（g/cm²）' , 'DXDTOBMD'),
        ('体脂率（%）' , 'DXDTOPF')
    ],
    'Food Security':[
        ('饮食是否充足' , 'FSD032B' , False , lambda x: 1 if x == 1 else 2 if x == 2 else 3 if x == 3 else None , '1 表示经常不足，2 表示偶尔不足，3 表示饮食充足'),
        ('饮食是否均衡' , 'FSD032C' , False , lambda x: False if x == 1 else True if x == 2 or x == 3 else None , 'False 表示基本单调，True 表示基本均衡'),
        ('是否会跳过正餐' , 'FSD041' , False , bool_check)
    ],
    'Glycohemoglobin':[],
    'Hepatitis':[
        (DISEASE_COLUMN , 'HEQ010' , True , append_disease('乙肝' , disease_check)),
        (DISEASE_COLUMN, 'HEQ030', True, append_disease('丙肝', disease_check))
    ],
    'High-Sensitivity C-Reactive Protein':[],
    'Housing Characteristics':[],
    'Insulin':[
        ('胰岛素（pmol/L）' , 'LBDINSI')
    ],
    'Iodine - Urine':[],
    'Kidney Conditions - Urology':[
        (DISEASE_COLUMN , 'KIQ022' , True , append_disease('肾衰竭' , disease_check)),
        (DISEASE_COLUMN , 'KIQ026' , True , append_disease('肾结石' , disease_check)),
        ('起夜次数' , 'KIQ480' , False , lambda x: x if x < 5 else None)
    ],
    'Medical Conditions':[
        disease('MCQ010' , '哮喘'),
        disease('MCQ053' , '贫血'),
        disease('MCQ160A' , '关节炎'),
        disease('MCQ160N' , '痛风'),
        disease('MCQ160B' , '心力衰竭'),
        disease('MCQ160C' , '冠心病'),
        disease('MCQ160D' , '心绞痛'),
        disease('MCQ160F' , '中风'),
        disease('MCQ160K' , '支气管炎'),
        disease('MCQ510A' , '脂肪肝'),
        disease('MCQ510C' , '肝硬化' , 3),
        disease('MCQ510D' , '肝炎' , 4),
        disease('MCQ550' , '胆结石')
    ],
    'Metals - Urine':[
        metal_urine('URXUBA' , '钡'),
        metal_urine('URXUCD' , '镉'),
        metal_urine('URXUCO' , '钴'),
        metal_urine('URXUCS' , '铯'),
        metal_urine('URXUMN' , '锰'),
        metal_urine('URXUMO' , '钼'),
        metal_urine('URXUPB' , '铅'),
        metal_urine('URXUSB' , '锑'),
        metal_urine('URXUSN' , '锡'),
        metal_urine('URXUTL' , '铊'),
        metal_urine('URXUTU' , '钨')
    ],
    'Oral Health':[],
    'Oral Health - Dentition':[],
    'Oral Health - Recommendation of Care':[
        disease('OHAROCDT' , '蛀牙')
    ],
    'Osteoporosis':[
        disease('OSQ060' , '骨质疏松'),
        ('父母是否患有骨质疏松' , 'OSQ150' , False , bool_check)
    ],
    'Physical Activity':[
        ('是否从事体力劳动' , 'PAQ605' , False , bool_check),
        ('是否每周进行进行中等以上强度锻炼' , ['PAQ650' , 'PAQ665'] , False , lambda x , y: True if x == True or y == True else False),
        ('平均每天中等以上强度锻炼时长（分钟）' , ['PAD660' , 'PAD675'] , False , lambda x , y: x + y),
        ('平均每天坐姿时长（分钟）' , 'PAD680')
    ],
    'Physical Activity - Youth':[],
    'Sex Steroid Hormone Panel - Serum (Surplus)':[],
    'Sleep Disorders':[
        ('工作日平均睡眠小时数' , 'SLD012'),
        ('打鼾频率' , 'SLQ030' , False , lambda x: x if x < 4 else None , '0=从不；1=很少（每周1-2晚）；2=偶尔（每周3-4晚）；3=频繁（每周5晚及以上）')
    ],
    'Smoking - Cigarette Use':[
        ('吸烟频率' , 'SMQ040' , False , lambda x: x if x < 4 else None , '1 为每天吸烟，2 为偶尔吸烟，3 为不吸烟。'),
        ('开始吸烟年龄' , 'SMD030')
    ],
    'Smoking - Household Smokers':[
        ('家中是否有人吸烟' , 'SMD470' , False , lambda x: True if x < 4 else None)
    ],
    'Smoking - Recent Tobacco Use':[],
    'Smoking - Secondhand Smoke Exposure':[
        ('一周内是否吸入过二手烟' , ['SMQ858' , 'SMQ862' , 'SMQ868' , 'SMQ872' , 'SMQ876' , 'SMQ880'] , False , lambda a , b , c , d , e , f: True if a == 1 or b == 1 or c == 1 or d == 1 or e == 1 or f == 1 else False)
    ],
    'Standard Biochemistry Profile':[
        ('葡萄糖（mg/dL）' , 'LBXSGL'),
        ('甘油三酯（mg/dL）' , 'LBXSTR'),
        ('肌酐（mg/dL）' , 'LBXSCR'),
        ('磷（mg/dL）' , 'LBXSLDSI'),
        ('钾（mmol/L）' , 'LBXSKSI'),
        ('白蛋白（g/dL）' , 'LBXSAL'),
        ('血尿素氮（mg/dL）' , 'LBXSBU'),
        ('碳酸氢盐（mmol/L）' , 'LBXSC3SI'),
        ('钠（mmol/L）' , 'LBXSNASI'),
        ('球蛋白（g/dL）' , 'LBXSGB')
    ],
    'Weight History':[],
    'Weight History - Youth':[],
}


def process() -> pandas.DataFrame:
    # 初始化结果集，选取任意一个表单作为索引来源，由于问卷覆盖的普遍性，该表单对结果集初始状态及后续数据处理影响不大。
    data_file = pandas.read_csv(DATA_PATH + "Dietary Interview - Total Nutrient Intakes, First Day.csv" , encoding = 'utf-8' , index_col='SEQN')
    result = pandas.DataFrame(index = data_file.index)
    result[DISEASE_COLUMN] = ''
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
                result[process[0]] = data_file[process[1]]
                continue
            if type(process[1]) is str:
                # 处理单数据项
                if process[2] is True:
                    # 处理单数据迭代
                    if process[0] not in result.columns:
                        result[process[0]] = None
                    result[process[0]] = result[process[0]].combine(data_file[process[1]] , process[3])
                else:
                    # 处理独立单数据项
                    if process[3] is None:
                        # 处理单数据转移
                        result[process[0]] = data_file[process[1]]
                    else:
                        # 处理独立单数据
                        result[process[0]] = data_file[process[1]].apply(process[3])
            else:
                # 处理多数据项
                columns = data_file[[*process[1]]]
                if process[2] is True:
                    # 处理多数据迭代
                    if process[0] not in result.columns:
                        result[process[0]] = None
                    columns.join(result[process[0]] , how = 'inner')
                    result[process[0]] = columns.apply(lambda row: process[3](*row) , axis = 1)
                else:
                    # 处理多数据
                    result[process[0]] = columns.apply(lambda row: process[3](*row) , axis = 1)
    return result

result = process()
result.to_csv(DATA_PATH[:-1] + ".csv" , encoding = 'utf-8')