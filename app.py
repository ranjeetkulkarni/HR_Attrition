from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
xgb_model = pickle.load(open('xgb_classifier.pkl', 'rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

# Load the scaler if it was saved; adjust path as necessary
# scaler = pickle.load(open('scaler.pkl', 'rb'))  # Uncomment if you have a saved scaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/hr/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    df_hr = pd.DataFrame([data])
    
    df_hr.drop(['EmployeeCount', 'StandardHours'], axis=1, inplace=True, errors='ignore')
    
    numerical_features = [feature for feature in df_hr.columns if df_hr[feature].dtypes != 'O']
    categorical_features = [
        'Attrition', 'BusinessTravel', 'Department', 'EducationField', 
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'
    ]
    
    year_feature = [
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    
    discrete_feature = [
        'Education', 'EmployeeCount', 'EnvironmentSatisfaction', 
        'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
        'StandardHours', 'StockOptionLevel', 'WorkLifeBalance'
    ]
    
    continuous_feature = [
        'EmployeeNumber', 'Age', 'DailyRate', 'DistanceFromHome', 
        'HourlyRate', 'MonthlyIncome', 'MonthlyRate'
    ]

    company_worked_mapping = {
        0: 0, 1: 1, 2: 3, 3: 3, 4: 3, 5: 6, 6: 6, 7: 6, 8: 9, 9: 9
    }

    # Apply the mapping to the 'NumCompaniesWorked' column
    df_hr['NumCompaniesWorked_Category'] = df_hr['NumCompaniesWorked'].map(company_worked_mapping)
    df_hr.drop('NumCompaniesWorked', axis=1, inplace=True, errors='ignore')
    df_hr.rename(columns={'NumCompaniesWorked_Category': 'NumCompaniesWorked'}, inplace=True)

    salary_hike_mapping = {
        11: 12, 12: 14, 13: 14, 14: 15, 15: 16,
        16: 18, 17: 18, 18: 18, 19: 18, 20: 21,
        21: 21, 22: 21, 23: 24, 24: 24, 25: 24
    }

    # Apply the mapping to the 'PercentSalaryHike' column
    df_hr['PercentSalaryHike_Category'] = df_hr['PercentSalaryHike'].map(salary_hike_mapping)
    df_hr.drop('PercentSalaryHike', axis=1, inplace=True, errors='ignore')
    df_hr.rename(columns={'PercentSalaryHike_Category': 'PercentSalaryHike'}, inplace=True)

    for feature in ['MonthlyIncome', 'DistanceFromHome']:
        df_hr[feature] = np.log1p(df_hr[feature])

    df_hr.drop(['EmployeeNumber', 'Over18', 'Gender', 'Department'], axis=1, inplace=True, errors='ignore')

    categorical_features = [
        'Attrition', 'BusinessTravel', 'EducationField', 'JobRole', 
        'MaritalStatus', 'OverTime'
    ]

    df_encoded = pd.get_dummies(df_hr, columns=[feature for feature in categorical_features if feature != 'Attrition'],
                                 drop_first=True)  # One-hot encode features

    df_encoded = df_encoded.replace({True: 1, False: 0})
    df_encoded['Attrition'] = df_encoded['Attrition'].replace({'Yes': 1, 'No': 0})

    numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

    # Apply scaling if you have a scaler loaded
    df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])  # Uncomment if scaler is available
    # Ensure the data is in array form for prediction
    data_for_prediction = df_encoded.values.reshape(1, -1) if df_encoded.shape[0] == 1 else df_encoded.values
    # Make prediction
    prediction = xgb_model.predict(data_for_prediction)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
     app.run(debug=True, port=5001)
