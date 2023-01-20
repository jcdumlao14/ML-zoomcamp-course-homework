
import requests

url = 'http://localhost:8888/predict'

patient_id ='2'
patient = {
    "age": 62.0,
    "sex": 0.0,
    "chest_pain_type": 4.0,
    "resting_bp_s": 120.0,
    "cholesterol": 0.0,
    "fasting_blood_sugar": 1.0,
    "resting_ecg": 1.0,
    "max_heart_rate": 123.0,
    "exercise_angina": 1.0,
    "oldpeak": 1.7,
    "st_slope": 3.0,
    
    }

# send this customer as a post request
response = requests.post(url,json=patient).json() 
print(response)

if response['Heart Disease'] == True:
    print("Patient %s who is suffering from heart risk" % patient_id)
else:
    print("Patient %s who is not suffering from heart risk" % patient_id)
    