import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()


@app.get('/')
def home():
    return{'text': 'Loan Default Prediction'}



class model_input(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    installment: float
    grade: int
    home_ownership: int
    annual_inc: float
    verification_status: int
    purpose: int
    dti: float
    open_acc:  float
    revol_bal: float
    total_acc: float
    mortage_acc:  float
    zip_code:  int



loan_model = pickle.load(open('loan_prediction_model', 'rb'))


@app.post('/loan_default_prediction')
def loan_pred(input_parameters: model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)


    loan_amnt = input_dictionary['loan_amnt']
    term = input_dictionary['term']
    int_rate = input_dictionary['int_rate']
    installment = input_dictionary['installment']
    grade = input_dictionary['grade']
    home_ownership = input_dictionary['home_ownership']
    annual_inc = input_dictionary['annual_inc']
    verification_status = input_dictionary['verification_status']
    purpose = input_dictionary['purpose']
    dti = input_dictionary['dti']
    open_acc = input_dictionary['open_acc']
    revol_bal = input_dictionary['revol_bal']
    total_acc = input_dictionary['total_acc']
    mort_acc = input_dictionary['mortage_acc']
    zip_code = input_dictionary['zip_code']


    input_list = [loan_amnt, term, int_rate, installment, grade, home_ownership,annual_inc, verification_status, purpose, dti, open_acc, revol_bal, total_acc,  mort_acc, zip_code]

    prediction = loan_model.predict([input_list])
    
    if prediction[0] == 0:
        return  'defaulter'
    
    else:
        return 'non_defaulter'

if __name__ == '__main_':
    uvicorn.run(app)


