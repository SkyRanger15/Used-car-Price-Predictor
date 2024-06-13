from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("LinRegmodel.pkl", "rb"))
car = pd.read_csv('Clean.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    companies.insert(0,'Select Brand')
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_type)

@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    selected_company = request.form['company']
    filtered_models = car[car['company'] == selected_company]['name'].unique().tolist()
    return jsonify(models=filtered_models)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')  # Correct field name
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))  # Ensure it is converted to int

    print(company, car_model, year, fuel_type, kms_driven)

    # Add prediction logic here
    # prediction = model.predict([[company, car_model, year, fuel_type, kms_driven]])  # Example placeholder
    prediction = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]], columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)
