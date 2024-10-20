from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the model
def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

def process_data(data):
    data = pd.DataFrame(data, index=[0])

    input_dict = {
        "CreditScore": data["CreditScore"],
        "Age": data["Age"],
        "Tenure": data["Tenure"],
        "Balance": data["Balance"],
        "NumOfProducts": data["NumOfProducts"],
        "HasCrCard": data["HasCrCard"],
        "IsActiveMember": data["IsActiveMember"],
        "EstimatedSalary": data["EstimatedSalary"],
        "Geography_France": 1 if data["Geography"] == "France" else 0,
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "Geography_Spain": 1 if data["Geography"] == "Spain" else 0,
        "Male": 1 if data["Gender"] == "Male" else 0,
        "CLV": data['Balance'] * data['EstimatedSalary'] / 10000,
        "TenureAgeRatio": data['Tenure'] / data['Age'],
        "AgeGroup_MiddleAge": 1 if data['Age'] >= 30 and data['Age'] < 45 else 0,
        "AgeGroup_Senior": 1 if data['Age'] >= 45 and data['Age'] < 60 else 0,
        "AgeGroup_Elderly": 1 if data['Age'] >= 60 else 0
    }

    input_df = pd.DataFrame(input_dict, index=[0])

    return input_df

def process_data_selective(data):
    data = pd.DataFrame(data, index=[0])

    input_dict = {
        "Age": data["Age"],
        "NumOfProducts": data["NumOfProducts"],
        "IsActiveMember": data["IsActiveMember"],
        "Geography_Germany": 1 if data["Geography"] == "Germany" else 0,
        "AgeGroup_Senior": 1 if data['Age'] >= 45 and data['Age'] < 60 else 0
    }

    input_df = pd.DataFrame(input_dict, index=[0])

    return input_df
    
def get_prediction(request):
    model_name = request['model']
    if model_name == 'xgb':
        model = load_model('xgb_smote.pkl')
    elif model_name == 'rf':
        model = load_model('rf_smote.pkl')
    elif model_name == 'gb-selective':
        model = load_model('gradientBoosting_selective.pkl')
    elif model_name == 'voting':
        model = load_model('voting_clf.pkl')
    elif model_name == 'stacking':
        model = load_model('stacking_smote.pkl')
    else:
        return "Invalid model name"

    if model_name == 'gb-selective':
        input_df = process_data_selective(request['data'])
    else:
        input_df = process_data(request['data'])
    
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]

    return prediction, probabilities

@app.post("/predict")
async def predict(request: dict):
    prediction, probabilities = get_prediction(request)

    return {
        "prediction": prediction[0],
        "probabilities": {
            "Exited": probabilities[0],
            "Not Exited": probabilities[1]
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)