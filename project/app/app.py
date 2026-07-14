from flask import Flask,request,jsonify
from pyexpat import features

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
    return jsonify({
        "service": "疾病预测系统",
        "version": "TEST",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/test', methods=['GET'])
def test():
    # 模拟一份假数据
    test_data = {
        "age": 45,
        "gender": 1,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "fasting_glucose": 5.8
    }
    # 直接调用你的核心预测函数
    is_valid, err = convert_and_validate(test_data)
    if is_valid:
        result = do_prediction(test_data)
        return jsonify({"status": "success", "result": result})
    else:
        return jsonify({"status": "error", "message": err})

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
