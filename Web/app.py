from flask import Flask, request, jsonify, render_template
from pyexpat import features
import json,os
from project.config import HOST,PORT,DEBUG
from business import convert_and_validate,do_prediction
#from business import do_prediction_demo

#请确保安装了flask_cors(如果需要)和flask

from flask_cors import CORS    #如果需要跨域支持，请取消注释


app = Flask(__name__)
CORS(app)          #如果需要跨域支持，请取消注释




################# API接口 ##################
@app.route('/predict', methods=['POST'])    #预测接口
def predict():      #接收前端数据并返回预测结果
    data = request.get_json()   #获取数据
    print(data)
    try:
        converted_data = convert_and_validate(data)  # ← 可能抛出异常
    except ValueError as e:
        # 通过接口返回错误信息
        return jsonify({'status': 'error', 'message': str(e)}), 400
    result = do_prediction(converted_data)    #进行预测(第一部分)
    return jsonify({'status':'success','result':result}),200    #返回预测结果

@app.route('/health', methods=['GET'])    #健康检查接口
def health_check():     #确认服务是否运行正常
    return jsonify({'status':'ok','message':'Service is up and running'}),200

@app.route('/', methods=['GET'])
def home():
    return render_template('6.html')

@app.route('/page_config', methods=['GET'])
def page_config():
    json_path = '../DataMiner/data/NHANES/2017_category.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return jsonify(config)

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
