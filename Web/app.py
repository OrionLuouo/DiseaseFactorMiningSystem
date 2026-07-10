from flask import app, render_template, Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

def initial():
    app.run(debug=True)

if __name__ == '__main__':
    initial()
else:
    print("Not running as main program")