# HR Attrition Prediction

## Problem Statement

**McCurr Consultancy**, an MNC, spends significant resources on retaining employees. The goal of this project is to identify factors contributing to employee attrition and build a predictive model to target at-risk employees with retention initiatives.

### Key Questions:
1. What factors can help identify employees at risk of attrition?  
2. Can a predictive model accurately classify employees likely to attrite?  
3. What evaluation metric is suitable for this use case?

---

## Dataset Overview

The dataset contains various features describing employees and their work environment. Key attributes include employee demographics, job details, satisfaction levels, and attrition status.  

### Attribute Information:

| **Feature**                | **Description**                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------|
| EmployeeNumber             | Unique employee identifier                                                                       |
| Attrition                  | Target variable: Whether the employee attrited (Yes/No)                                         |
| Age                        | Employee's age                                                                                  |
| BusinessTravel             | Frequency of business travel                                                                    |
| DailyRate                  | No description provided                                                                         |
| Department                 | Employee's department                                                                           |
| DistanceFromHome           | Distance from work to home (in km)                                                              |
| Education                  | Education level (1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor)               |
| EducationField             | Field of education                                                                              |
| EnvironmentSatisfaction    | Satisfaction with work environment (1: Low, 2: Medium, 3: High, 4: Very High)                   |
| JobInvolvement             | Job involvement level (1: Low, 2: Medium, 3: High, 4: Very High)                                |
| JobLevel                   | Level of job (1 to 5)                                                                           |
| JobSatisfaction            | Job satisfaction level (1: Low, 2: Medium, 3: High, 4: Very High)                               |
| MaritalStatus              | Marital status of the employee                                                                  |
| MonthlyIncome              | Monthly salary                                                                                  |
| NumCompaniesWorked         | Number of companies the employee has worked at                                                 |
| Over18                     | Whether the employee is over 18 years old                                                      |
| OverTime                   | Whether the employee works overtime                                                            |
| PercentSalaryHike          | Percentage increase in salary last year                                                        |
| PerformanceRating          | Employee performance rating (1: Low, 2: Good, 3: Excellent, 4: Outstanding)                    |
| RelationshipSatisfaction   | Relationship satisfaction level (1: Low, 2: Medium, 3: High, 4: Very High)                      |
| TotalWorkingYears          | Total years the employee has worked                                                             |
| TrainingTimesLastYear      | Number of training sessions attended last year                                                 |
| WorkLifeBalance            | Work-life balance level (1: Low, 2: Good, 3: Excellent, 4: Outstanding)                         |
| YearsAtCompany             | Number of years the employee has been with the company                                          |
| YearsInCurrentRole         | Number of years the employee has been in their current role                                     |
| YearsSinceLastPromotion    | Number of years since the employee's last promotion                                             |
| YearsWithCurrManager       | Number of years the employee has been with their current manager                                |

---

## Learning Outcomes

Through this project, you will:
- Perform **Exploratory Data Analysis (EDA)** to uncover key patterns in the data.
- Prepare the dataset for modeling by handling missing values, encoding categorical variables, and scaling.
- Train predictive models, including **Decision Tree** and **Ensemble Models**.
- Evaluate the models using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Provide actionable insights and conclusions based on the model's results.

---

## Steps and Tasks

1. **Import Libraries and Load Dataset**: Import necessary libraries like pandas, matplotlib, and scikit-learn, and load the dataset.
2. **Overview of Data**: Understand the dataset structure, check for missing values, and explore summary statistics.
3. **Data Visualization**: Create visualizations to identify key trends and relationships between variables.
4. **Data Preparation**: Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
5. **Choose Model, Train, and Evaluate**: Train models like Decision Trees and Ensemble Models, and evaluate their performance using appropriate metrics.
6. **Conclusion**: Summarize findings, key factors influencing attrition, and the best-performing model.

---

## Technologies Used

- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook
- Machine Learning algorithms (Decision Trees, Random Forest, etc.)

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ranjeetkulkarni/hrattrition.git
