import pandas

def func(a , b):
    return a + b

data_file = pandas.read_csv('../../data/NHANES/2017/Alcohol Use.csv' , encoding='utf-8' , index_col='SEQN')
result = pandas.DataFrame()
result.join(data_file[['ALQ111']] , how = 'inner').join(data_file[['ALQ121' , 'ALQ130']] , how='inner')

result['qwq'] = func(result['ALQ111'], result['ALQ121'])

result.to_csv('test.csv')