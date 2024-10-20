# Customer Churn Prediction API

## Overview
The Customer Churn Prediction API provides machine learning models that predict whether a bank customer is likely to churn. By leveraging various algorithms, the API enables businesses to make informed decisions and take proactive measures to retain valuable customers.

## Customer Data Requirements
To make a prediction, the API requires the following customer data fields:

- **CreditScore**: (integer) The credit score of the customer.
- **Geography**: (string) The country of residence (e.g., "Spain").
- **Gender**: (string) The gender of the customer (e.g., "Male" or "Female").
- **Age**: (integer) The age of the customer.
- **Tenure**: (integer) The number of years the customer has been with the bank.
- **Balance**: (float) The current account balance of the customer.
- **NumOfProducts**: (integer) The number of products the customer has with the bank.
- **HasCrCard**: (integer) Indicates if the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: (integer) Indicates if the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: (float) The estimated salary of the customer.

## Available Models
The API supports the following machine learning models for churn prediction:

- **xgb**: XGBoost model.
- **rf**: Random Forest model.
- **gb-selective**: Selective Gradient Boosting model.
- **voting**: Voting ensemble model.
- **stacking**: Stacking ensemble model.

## API Usage
To use the API, send a POST request to the `/predict` endpoint with the appropriate data format. Below is an example of how to test the API using Python:

### Example Code
```python
import requests
import json

# URL of deployed model
url = "https://churn-ml-models-b1ch.onrender.com/predict"

# Sample customer data
customer_data = {
    "CreditScore": 519,
    "Geography": "Spain",
    "Gender": "Male",
    "Age": 52,
    "Tenure": 2,
    "Balance": 0.00,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.82
}

models = ["xgb", "rf", "gb-selective", "voting", "stacking"]
model = models[0]  # Select a model

test_data = {
    "model": model,
    "data": customer_data
}

# Send POST request
response = requests.post(url, json=test_data)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
