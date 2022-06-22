
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__)
model = pickle.load(open('modell.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('ind.html')

@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    return render_template('ind.html', prediction_text='Status :{}'.format(prediction))


if __name__ == "__main__":
    application.run(debug=True)