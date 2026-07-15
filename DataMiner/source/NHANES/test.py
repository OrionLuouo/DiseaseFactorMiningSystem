import pandas

data_file = pandas.read_csv('../../data/NHANES/2017_cleaned.csv' , encoding = 'utf-8')

print(len(data_file.columns))