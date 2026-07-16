import pandas

data_file = pandas.read_csv('../../data/NHANES/2017.csv' , encoding = 'utf-8')

print(data_file)

data_file = pandas.read_csv('../../data/NHANES/2017_cleaned.csv' , encoding = 'utf-8')

print(data_file)