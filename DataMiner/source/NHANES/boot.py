from requests.exceptions import SSLError

import get_data
import process_data
import standardize_data

# 数据处理一键式脚本
try:
    get_data.run()
except SSLError as ssl_error:
    print("网络错误！无法从源网站采集数据。")
process_data.run()
standardize_data.run()