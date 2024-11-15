import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\home_loan.pkl', 'rb') as f:
    home_loan_model = pickle.load(f)  

with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\scaling_params.pkl', 'rb') as f:
    standard_params = pickle.load(f)  
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\other_loan.pkl', 'rb') as f:
    other_loan_model = pickle.load(f)
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\scaling_params_other.pkl', 'rb') as f:
    scaling_params_other = pickle.load(f)
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\personal_loan.pkl', 'rb') as f:
    personal_loan_model = pickle.load(f)
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\scaling_params_personal.pkl', 'rb') as f:
    scaling_params_personal = pickle.load(f)
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\vehicle_loan.pkl', 'rb') as f:
    vehicle_loan_model = pickle.load(f)
with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\scaling_params_vehicle.pkl', 'rb') as f:
    scaling_params_vehicle = pickle.load(f)
x_mean = standard_params['x_mean']
x_std = standard_params['x_std']
x_mean_p = scaling_params_personal['x_mean']
x_std_p = scaling_params_personal['x_std']
x_mean_v = scaling_params_vehicle['x_mean']
x_std_v = scaling_params_vehicle['x_std']
def preprocess_input(xam):
    for i in range(len(xam)):
        if i == 0:  
            xam[i] = 1 if xam[i] == "Male" else 0
        elif i == 1:  
            xam[i] = 1 if xam[i] == "Yes" else 0
        elif i == 2:  
            xam[i] = 0 if xam[i] in ["0", 0] else 1 if xam[i] in ["1", 1] else 2 if xam[i] in ["2", 2] else 3
        elif i == 3:  
            xam[i] = 1 if xam[i] == "Graduate" else 0
        elif i == 4:  
            xam[i] = 1 if xam[i] == "Yes" else 0
        elif i == 5:  
            xam[i] = (float(xam[i]) - x_mean[0]) / x_std[0]
        elif i == 6:  
            xam[i] = (float(xam[i]) - x_mean[1]) / x_std[1]
        elif i == 7:  
            xam[i] = (float(xam[i]) - x_mean[2]) / x_std[2]
        elif i == 8:  
            xam[i] = (float(xam[i]) - x_mean[3]) / x_std[3]
        elif i == 9:  
            xam[i] = 0 if xam[i] in ["0", 0] else 1 if xam[i] in ["1", 1] else 3
        elif i == 10:  
            xam[i] = 1 if xam[i] == "Semiurban" else 2 if xam[i] == "Urban" else 0
        elif i == 11:  
            xam[i] = 1 if xam[i] == "Y" else 0
    return np.array(xam).reshape(1, -1)
def preprocess_input_per(xam):
    for i in range(len(xam)):
        if i==0:
            xam[i]=xam[i]=(xam[i]-x_mean_p[0])/x_std_p[0]
        if i==1:
            xam[i]=xam[i]=(xam[i]-x_mean_p[1])/x_std_p[1]
        if i==2:
            xam[i]=xam[i]=(xam[i]-x_mean_p[2])/x_std_p[2]
        if  i==4:
            xam[i]=xam[i]=(xam[i]-x_mean_p[3])/x_std_p[3]
        if i==6:
            xam[i]=xam[i]=(xam[i]-x_mean_p[4])/x_std_p[4]
    return np.array(xam).reshape(1, -1)
def preprocess_input_vehicle(xam):
    data = pd.read_csv("C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\credit_data.csv")


    state_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    employment_encoder = LabelEncoder()


    data['State_Encoded'] = state_encoder.fit_transform(data['State'])
    data['City_Encoded'] = city_encoder.fit_transform(data['City'])
    data['Employment_Type_Encoded'] = employment_encoder.fit_transform(data['Employment Profile'])


    state_mapping = dict(zip(state_encoder.classes_, state_encoder.transform(state_encoder.classes_)))
    city_mapping = dict(zip(city_encoder.classes_, city_encoder.transform(city_encoder.classes_)))
    employment_mapping = dict(zip(employment_encoder.classes_, employment_encoder.transform(employment_encoder.classes_)))
    for i in range(len(xam)):
        if i == 0:  
            xam[i] = (xam[i] - x_mean_v[0]) / x_std_v[0]

        elif i == 1:  
            xam[i] = 0 if xam[i] == "Male" else 1 if xam[i] == "Female" else 2

        elif i == 2:  
            xam[i] = (xam[i] - x_mean_v[1]) / x_std_v[1]

        elif i == 3:  
            xam[i] = (xam[i] - x_mean_v[2]) / x_std_v[2]

        elif i == 4:  
            xam[i] = (xam[i] - x_mean_v[3]) / x_std_v[3]

        elif i == 5:  
            xam[i] = (xam[i] - x_mean_v[4]) / x_std_v[4]

        elif i == 6:  
            xam[i] = (xam[i] - x_mean_v[5]) / x_std_v[5]

        elif i == 7:  
            xam[i] = (xam[i] - x_mean_v[6]) / x_std_v[6]

        elif i == 8:  
            xam[i] = 1 if xam[i] == "Yes" else 0

        elif i == 9:  
            xam[i] = state_mapping.get(xam[i], -1)

        elif i == 10:  
            xam[i] = city_mapping.get(xam[i], -1)

        elif i == 11:  
            xam[i] = (xam[i] - x_mean_v[7]) / x_std_v[7]

        elif i == 12:  
            xam[i] = employment_mapping.get(xam[i], -1)
    return np.array(xam).reshape(1, -1)

def sf(z):
    return 1 / (1 + np.exp(-z))


def hf(w, x):
    return sf(x.dot(w)) >= 0.5


def custom_predict(input_data, best_w):
    
    input_data = np.c_[np.ones(input_data.shape[0]), input_data]  
    return hf(best_w, input_data)  


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home_loan', methods=['GET', 'POST'])
def home_loan():
    if request.method == 'POST':
        
        xam = [
            request.form['gender'],
            request.form['married'],
            request.form['dependents'],
            request.form['education'],
            request.form['self_employed'],
            request.form['applicant_income'],
            request.form['coapplicant_income'],
            request.form['loan_amount'],
            request.form['loan_amount_term'],
            request.form['credit_history'],
            request.form['property_area']
        ]
        
        
        standardized_input = preprocess_input(xam)
        
        try:
            
            prediction = home_loan_model.predict(standardized_input)
            result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
            return render_template('home_loan.html', result=result, model='Random Forest')

        except Exception as e:
            
            print(f"Error with Home Loan Model: {str(e)}")  
            try:
                
                with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\home_loan.pkl', 'rb') as f:
                    best_w = pickle.load(f)  
                
                
                custom_prediction = custom_predict(standardized_input, best_w)
                result = 'Eligible' if custom_prediction[0] else 'Not Eligible'
                return render_template('home_loan.html', result=result, model='Custom Logistic Regression')

            except Exception as e:
                
                print(f"Error with Custom Model: {str(e)}")  
                return render_template('home_loan.html', error="Prediction failed for both models. Please check input data.")

    return render_template('home_loan.html')
@app.route("/other_loan", methods=["GET", "POST"])
def other_loan():
    if request.method == 'POST':
        
        xam = [
            request.form['gender'],
            request.form['married'],
            request.form['dependents'],
            request.form['education'],
            request.form['self_employed'],
            request.form['applicant_income'],
            request.form['coapplicant_income'],
            request.form['loan_amount'],
            request.form['loan_amount_term'],
            request.form['credit_history'],
            request.form['property_area']
        ]
        
        
        standardized_input = preprocess_input(xam)
        
        try:
            
            prediction = other_loan_model.predict(standardized_input)
            result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
            return render_template('other_loan.html', result=result, model='Random Forest')

        except Exception as e:
            
            print(f"Error with Home Loan Model: {str(e)}")  
            try:
                
                with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\other_loan.pkl', 'rb') as f:
                    best_w = pickle.load(f)  
                
                
                custom_prediction = custom_predict(standardized_input, best_w)
                result = 'Eligible' if custom_prediction[0] else 'Not Eligible'
                return render_template('other_loan.html', result=result, model='Custom Logistic Regression')

            except Exception as e:
                
                print(f"Error with Custom Model: {str(e)}")  
                return render_template('other_loan.html', error="Prediction failed for both models. Please check input data.")

    return render_template('other_loan.html')
@app.route('/personal_loan', methods=['GET', 'POST'])
def personal_loan():
    if request.method == 'POST':
        
        try:
            xam = [
                float(request.form['age']),
                float(request.form['experience']),
                float(request.form['income']),
                float(request.form['family']),
                float(request.form['ccavg']),
                float(request.form['education']),
                float(request.form['mortgage']),
                float(request.form['securities_account']),
                float(request.form['cd_account']),
                float(request.form['online']),
                float(request.form['credit_card'])
            ]
        except ValueError as e:
            return render_template('personal_loan.html', error="Invalid input: please enter numeric values only.")

        
        standardized_input = preprocess_input_per(xam)

        try:
            
            prediction = personal_loan_model.predict(standardized_input)
            result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
            return render_template('personal_loan.html', result=result, model='Random Forest')

        except Exception as e:
            
            print(f"Error with Personal Loan Model: {str(e)}")
            try:
                with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\personal_loan.pkl', 'rb') as f:
                    best_w = pickle.load(f)
                
                custom_prediction = custom_predict(standardized_input, best_w)
                result = 'Eligible' if custom_prediction[0] else 'Not Eligible'
                return render_template('personal_loan.html', result=result, model='Custom Logistic Regression')

            except Exception as e:
                print(f"Error with Custom Model: {str(e)}")
                return render_template('personal_loan.html', error="Prediction failed for both models. Please check input data.")

    return render_template('personal_loan.html')

@app.route("/vehicle_loan", methods=["GET", "POST"])
def vehicle_loan():
    if request.method == 'POST':
        # Collect form data
        xam = [
            float(request.form['age']),
            request.form['gender'],
            float(request.form['income']),
            float(request.form['credit_score']),
            float(request.form['credit_history_length']),
            int(request.form['existing_loans']),
            float(request.form['loan_amount']),
            int(request.form['loan_tenure']),
            request.form['existing_customer'],
            request.form['state'],
            request.form['city'],
            float(request.form['ltv_ratio']),
            request.form['employment_profile']
        ]

        standardized_input = preprocess_input_vehicle(xam)

        try:
            
            prediction = vehicle_loan_model.predict(standardized_input)
            result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
            return render_template('vehicle_loan.html', result=result, model='Random Forest')

        except Exception as e:
            
            print(f"Error with Personal Loan Model: {str(e)}")
            try:
                with open('C:\\Users\\satya\\OneDrive\\Documents\\ml_project\\website\\models\\vehicle_loan.pkl', 'rb') as f:
                    best_w = pickle.load(f)
                
                custom_prediction = custom_predict(standardized_input, best_w)
                result = 'Eligible' if custom_prediction[0] else 'Not Eligible'
                return render_template('vehicle_loan.html', result=result, model='Custom Logistic Regression')

            except Exception as e:
                print(f"Error with Custom Model: {str(e)}")
                return render_template('vehicle_loan.html', error="Prediction failed for both models. Please check input data.")

    return render_template('vehicle_loan.html')


if __name__ == '__main__':
    app.run(debug=True)
