import pickle
from flask import Flask, request, app, redirect, url_for, render_template, jsonify
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('logisticreg.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """here we get the predict data so we have to scale the data first and convert it into 2D array"""
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    predict_data = np.array(list(data.values())).reshape(1,-1)
    new_data = scalar.transform(predict_data)
    output = model.pridict(new_data)
    print(output[0])

    return jsonify(output[0])


@app.route("/predict", methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.pridict(final_input)[0]

    return render_template("home.html", prediction_text="the predict value is {}".format(output))

if __name__ == "__main__":

    app.run(debug=True)