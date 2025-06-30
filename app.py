import pickle
from flask import  Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the Model
regmodel = pickle.load(open("logisticreg.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("data========",data)
    print("erray=========",np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print("output============",output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print("data****************",data)
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                        'JobRole', 'MaritalStatus', 'Over18', 'OverTime']

    numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
                        'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
                        'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                        'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
                        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                        'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    encoded_cats = []
    for col in categorical_features:
        encoder = encoders[col]
        val = data[col]
        encoded_val = encoder.transform([val])[0]
        encoded_cats.append(encoded_val)

    numerical_vals = [float(data[col]) for col in numerical_features]
    final_input = np.array(encoded_cats + numerical_vals).reshape(1, -1)
    

    scaled_input  = scaler.transform(final_input)
    print("final input =====", scaled_input )
    output = regmodel.predict(scaled_input )[0]
    predicted_label = "Yes" if output == 1 else "No"

    return render_template("home.html", prediction_text="The prediction is employee leave the company- {}".format(predicted_label))

if __name__=="__main__":
    app.run(debug=True)