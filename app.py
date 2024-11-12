from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scaler with error handling
try:
    xgb_model = pickle.load(open('xgb_classifier.pkl', 'rb'))
    scaler = pickle.load(open('scaling.pkl', 'rb'))
except FileNotFoundError as e:
    logging.error(f"Model or scaler file not found: {e}")
    raise

# Function to preprocess data
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(['EmployeeCount', 'StandardHours'], axis=1, inplace=True, errors='ignore')
    
    # Feature transformations
    company_worked_mapping = {0: 0, 1: 1, 2: 3, 3: 3, 4: 3, 5: 6, 6: 6, 7: 6, 8: 9, 9: 9}
    df['NumCompaniesWorked_Category'] = df['NumCompaniesWorked'].map(company_worked_mapping)
    df.drop('NumCompaniesWorked', axis=1, inplace=True, errors='ignore')
    df.rename(columns={'NumCompaniesWorked_Category': 'NumCompaniesWorked'}, inplace=True)

    salary_hike_mapping = {11: 12, 12: 14, 13: 14, 14: 15, 15: 16, 16: 18, 17: 18, 18: 18, 
                           19: 18, 20: 21, 21: 21, 22: 21, 23: 24, 24: 24, 25: 24}
    df['PercentSalaryHike_Category'] = df['PercentSalaryHike'].map(salary_hike_mapping)
    df.drop('PercentSalaryHike', axis=1, inplace=True, errors='ignore')
    df.rename(columns={'PercentSalaryHike_Category': 'PercentSalaryHike'}, inplace=True)

    # Log transformation for specific features
    for feature in ['MonthlyIncome', 'DistanceFromHome']:
        df[feature] = np.log1p(df[feature])

    df.drop(['EmployeeNumber', 'Over18', 'Gender', 'Department'], axis=1, inplace=True, errors='ignore')

    # One-hot encoding for categorical features
    categorical_features = ['BusinessTravel', 'EducationField', 'JobRole', 'MaritalStatus', 'OverTime']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Apply scaling
    numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])

    return df_encoded

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'EmployeeNumber': int(request.form.get('EmployeeNumber')),
            'Age': int(request.form.get('Age')),
            'BusinessTravel': request.form.get('BusinessTravel'),
            'DailyRate': int(request.form.get('DailyRate')),
            'Department': request.form.get('Department'),
            'DistanceFromHome': int(request.form.get('DistanceFromHome')),
            'Education': int(request.form.get('Education')),
            'EducationField': request.form.get('EducationField'),
            'EmployeeCount': int(request.form.get('EmployeeCount')),
            'EnvironmentSatisfaction': int(request.form.get('EnvironmentSatisfaction')),
            'Gender': request.form.get('Gender'),
            'HourlyRate': int(request.form.get('HourlyRate')),
            'JobInvolvement': int(request.form.get('JobInvolvement')),
            'JobLevel': int(request.form.get('JobLevel')),
            'JobRole': request.form.get('JobRole'),
            'JobSatisfaction': int(request.form.get('JobSatisfaction')),
            'MaritalStatus': request.form.get('MaritalStatus'),
            'MonthlyIncome': int(request.form.get('MonthlyIncome')),
            'MonthlyRate': int(request.form.get('MonthlyRate')),
            'NumCompaniesWorked': int(request.form.get('NumCompaniesWorked')),
            'Over18': request.form.get('Over18'),
            'OverTime': request.form.get('OverTime'),
            'PercentSalaryHike': int(request.form.get('PercentSalaryHike')),
            'PerformanceRating': int(request.form.get('PerformanceRating')),
            'RelationshipSatisfaction': int(request.form.get('RelationshipSatisfaction')),
            'StandardHours': int(request.form.get('StandardHours')),
            'StockOptionLevel': int(request.form.get('StockOptionLevel')),
            'TotalWorkingYears': int(request.form.get('TotalWorkingYears')),
            'TrainingTimesLastYear': int(request.form.get('TrainingTimesLastYear')),
            'WorkLifeBalance': int(request.form.get('WorkLifeBalance')),
            'YearsAtCompany': int(request.form.get('YearsAtCompany')),
            'YearsInCurrentRole': int(request.form.get('YearsInCurrentRole')),
            'YearsSinceLastPromotion': int(request.form.get('YearsSinceLastPromotion')),
            'YearsWithCurrManager': int(request.form.get('YearsWithCurrManager'))
        }

        df_hr = pd.DataFrame([input_data])
        df_encoded = preprocess_data(df_hr)

        # Make prediction
        prediction = xgb_model.predict(df_encoded.values.reshape(1, -1))
        prediction_label = "Yes" if prediction[0] == 1 else "No"
        return render_template('home.html', prediction=prediction_label)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('home.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
