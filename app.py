from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/submitImage', methods=['POST'])
def submitImage():
    print(request.form)
