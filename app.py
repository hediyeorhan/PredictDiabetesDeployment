import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = list()
    for i in request.form.values():
        features.append(i)
    
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)
    
    return render_template('index.html',prediction_text='Seker hastası olma ihtimaliniz  % {}'.format(prediction[0][0]*100))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)