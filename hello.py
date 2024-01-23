from model import ModelPipeline
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from flask import jsonify
import joblib
import os

REG_MODEL = joblib.load(f'{os.getcwd()}/RandomForest_regression.joblib')
CLASS_MODEL = joblib.load(f'{os.getcwd()}/rnd_clf_THIS_MODEL.joblib')
MODEL_PIPE = ModelPipeline(REG_MODEL, CLASS_MODEL)

application = Flask(__name__)

@application.route('/', methods=['GET', 'POST'])
def home_page():
    resp = ''
    if request.method == 'POST':
        exmple = [
            {
                'HB': float(request.form['HB']),
                'Ultimate_strength': float(request.form['ultimate_strength']),
                'E': float(request.form['E']),
                'ro': float(request.form['ro']),
                'c': float(request.form['c'])
            }
        ] 
        y = MODEL_PIPE(exmple)[0]
        output = ''
        for k in y.keys():
            output += k + f'<sub>{y[k]}</sub>'
        
        resp = f'<div class="col" id="response"><p>Ваш сплав:</p><p>{output}</p></div>'

    return render_template('index_ru.html', resp=resp)


@application.route('/api')
def api_page():
    return render_template('api_ru.html')


@application.route('/about')
def about_page():
    return render_template('about_ru.html')


@application.route('/predict', methods=['POST'])
def predict_materia():
    data = request.get_json()
    response = MODEL_PIPE(data)
    return jsonify(response), 200, {'ContentType':'application/json'}


@application.errorhandler(404)
def not_found(error):
    return 'Страница не найдена', 404


if __name__ == '__main__':
    application.run(debug=True)