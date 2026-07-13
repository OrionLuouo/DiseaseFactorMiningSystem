import pandas

data_file = pandas.read_csv('../../data/NHANES/2017/Alcohol Use.csv' , index_col = 0 , encoding = 'utf-8')

new_file = pandas.DataFrame(index = data_file.index)


# 布尔值检查，1 = True，2 = False，None = None
def bool_check(x:int):
    if x is None:
        return None
    if x > 2:
        return None
    return x == 1

new_file = new_file.join(data_file['ALQ111'] , how = 'inner')

new_file['bool'] = data_file['ALQ111'].apply(bool_check)

print(new_file)