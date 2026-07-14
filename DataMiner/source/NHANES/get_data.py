import os
from io import BytesIO

import pandas
import requests
from bs4 import BeautifulSoup, Tag, NavigableString

# 所需数据的名称。
DATA_NAMES = {'Demographics Data' , 'Dietary Data' , 'Examination Data' , 'Laboratory Data' , 'Questionnaire Data'}

# 每类数据下所需表的映射关系
SHEET_MAP = {
    'Demographics Data':['Demographic Variables and Sample Weights'],
    'Dietary Data': ['Dietary Interview - Total Nutrient Intakes, First Day'],
    'Examination Data': ['Blood Pressure' , 'Body Measures' , 'Dual-Energy X-ray Absorptiometry - Whole Body' , 'Oral Health - Dentition' , 'Oral Health - Recommendation of Care'],
    'Laboratory Data': ['Cholesterol - Total' , 'Complete Blood Count with 5-Part Differential' , 'Glycohemoglobin' , 'High-Sensitivity C-Reactive Protein' , 'Insulin' , 'Iodine - Urine' , 'Metals - Urine' , 'Plasma Fasting Glucose' , 'Sex Steroid Hormone Panel - Serum (Surplus)' , 'Standard Biochemistry Profile'],
    'Questionnaire Data': ['Alcohol Use' , 'Audiometry' , 'Blood Pressure & Cholesterol' , 'Cardiovascular Health' , 'Current Health Status' , 'Dermatology' , 'Diabetes' , 'Diet Behavior & Nutrition' , 'Food Security' , 'Hepatitis' , 'Housing Characteristics' , 'Kidney Conditions - Urology' , 'Medical Conditions' , 'Oral Health' , 'Osteoporosis' , 'Physical Activity' , 'Physical Activity - Youth' , 'Sleep Disorders' , 'Smoking - Cigarette Use' , 'Smoking - Household Smokers' , 'Smoking - Recent Tobacco Use' , 'Smoking - Secondhand Smoke Exposure' , 'Weight History' , 'Weight History - Youth']
}

# 网站主页网址。
ORIGIN = 'https://wwwn.cdc.gov/'

# 期刊列表网址。
MAIN_HERF = ORIGIN + 'nchs/nhanes/continuousnhanes/'

# 期望获取的期刊起始年限。
TARGET_YEAR = '2017'

# NHANES 调查期刊的链接，具体到年限，如 https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017。
SURVEY_HERF = MAIN_HERF + 'default.aspx?BeginYear=2017'

DATA_PATH = '../../data/NHANES/' + TARGET_YEAR + '/'

OVERRIDDEN_OLD_DATA = False

# 搜索期刊页面，寻找所需数据页连接。
def search_survey():
    HEADER = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'max-age=0',
        'cookie': 'akamai_visit_id=4c52cd17870739e08dbc; _cdc_vid=eb6ff82d-c2a9-45f5-b04c-83ff57089631; _ga=GA1.1.1391387165.1783687077; ASLBSA=000354139e6f28573001ff78d7c4ccda170f8feda2244123eec9a546a7f1d1d5673f; ASLBSACORS=000354139e6f28573001ff78d7c4ccda170f8feda2244123eec9a546a7f1d1d5673f; _cdc_test_c=value; _cdc_vnum=2; _cdc_sid=457193ff-46a1-47ba-94bf-b482560acc44; _cdc_ppn=NHANES%2520Questionnaires%252C%2520Datasets%252C%2520and%2520Related%2520Documentation; _cdc_occ=4; _cdc_ppu=https%3A%2F%2Fwwwn.cdc.gov%2Fnchs%2Fnhanes%2Fcontinuousnhanes%2Fdefault.aspx%3FBeginYear%3D2017; _ga_CSLL4ZEK4L=GS2.1.s1783753957$o2$g1$t1783753974$j43$l0$h0',
        'priority': 'u=0, i',
        'referer': 'https://wwwn.cdc.gov/nchs/nhanes/default.aspx',
        'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'
    }
    # 调查期刊的主页面
    main_page = requests.get(SURVEY_HERF , headers = HEADER , verify = False)
    # 使用 BeautifulSoup 获取元素内容。
    soup = BeautifulSoup(main_page.content , 'lxml')
    tags = soup.find_all('a' , class_ = 'list-title td-none td-ul-hover')
    for tag in tags:
        texts = tag.find_all(string = True , recursive = False)
        for text in texts:
            if text in DATA_NAMES:
                get_data_list_of(tag, text)

# 在数据页获取数据列表
def get_data_list_of(tag: Tag, text: NavigableString):
    HEADER = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'max-age=0',
        'cookie': 'akamai_visit_id=4c52cd17870739e08dbc; _cdc_vid=eb6ff82d-c2a9-45f5-b04c-83ff57089631; _ga=GA1.1.1391387165.1783687077; _cdc_vnum=2; _cdc_sid=457193ff-46a1-47ba-94bf-b482560acc44; ASLBSA=000354139e6f28573001ff78d7c4ccda170f8feda2244123eec9a546a7f1d1d5673f; ASLBSACORS=000354139e6f28573001ff78d7c4ccda170f8feda2244123eec9a546a7f1d1d5673f; _cdc_test_c=value; _cdc_occ=10; _cdc_ppu=https%3A%2F%2Fwwwn.cdc.gov%2Fnchs%2Fnhanes%2Fsearch%2Fdatapage.aspx%3FComponent%3DDietary%26CycleBeginYear%3D2017; _cdc_ppn=2017-2018%2520Dietary%2520Data%2520-%2520Continuous%2520NHANES; _ga_CSLL4ZEK4L=GS2.1.s1783753957$o2$g1$t1783758012$j56$l0$h0',
        'priority': 'u=0, i',
        'referer': 'https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017',
        'sec-ch-ua': '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36'
    }
    if not text in SHEET_MAP:
        print('> Warning: Data type "' , text , '" is not in SHEET_MAP.' , sep = '')
        return
    LIST = SHEET_MAP[text]
    print('> Reading list of type "' , text , '": ' , LIST , sep = '')
    href = tag.get('href')
    if type(href) is not str:
        return
    data_href = MAIN_HERF + href
    data_page = requests.get(data_href , headers = HEADER , verify = False)
    soup = BeautifulSoup(data_page.content , 'lxml')
    root_tag = soup.find('table', class_='table table-bordered table-header-light table-striped table-hover table-hover-light nein-scroll')
    trs = root_tag.find_all('tr')
    for tr in trs:
        td = tr.find('td' , class_ = 'text-left')
        if td is None:
            continue
        name = td.find(string = True , recursive = False)
        if name is None:
            continue
        name:str = name.text.strip()
        if name in LIST:
            get_data(tr , name)

# 获取数据表
def get_data(tr: Tag , name: str):
    path = DATA_PATH + name + '.csv'
    if os.path.exists(path):
        if not OVERRIDDEN_OLD_DATA:
            print(' > Data sheet "', name, '" already exists.', sep='')
            return
        os.remove(path)
    hrefs = tr.find_all('a')
    for href_tag in hrefs:
        href = href_tag.get('href')
        if type(href) is not str:
            continue
        if href.endswith('xpt'):
            href = ORIGIN[:-1] + href
            data_file = requests.get(href , verify = False)
            data_file = BytesIO(data_file.content)
            data_file = pandas.read_sas(data_file , encoding = 'utf-8' , format = 'xport')
            data_file.to_csv(path , index = False , encoding = 'utf-8')
            print(' > Data sheet "' , name , '" has been downloaded to ' , path , '.' , sep = '')

def run():
    print("> Getting data...")
    os.makedirs(DATA_PATH , exist_ok = True)
    search_survey()

if __name__ == '__main__':
    run()