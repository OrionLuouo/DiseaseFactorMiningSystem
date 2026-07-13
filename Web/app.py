from flask import app, render_template, Flask

app = Flask(__name__)

@app.route('/get_table')
def get_table():
    # ... 从存储中获取报表
    return ('{"rows":['
            '"身高","体重","血压","血脂"'
            ']}')

@app.route('/')
def hello():
    return render_template('index.html')

def initial():
    app.run(debug=True)

if __name__ == '__main__':
    initial()
else:
    print("Not running as main program")