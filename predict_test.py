import requests

url = "http://localhost:9696/predict"


patient = {
    "Age": 46,
    "Sex": "M",                          
    "ChestPainType": "ATA",              
    "RestingBP": 167,                    
    "Cholesterol": 163,                  
    "FastingBS": 0,                      
    "RestingECG": "ST",                  
    "MaxHR": 103,                        
    "ExerciseAngina": "N",               
    "Oldpeak": 1.5,                      
    "ST_Slope": "Down"                   
}

response = requests.post(url, json = patient).json()

print(response)

if response['heart_disease']:
    print('Define a treatment for the patient-test.')
else:
    print('The patient seems healthy: no treatment needed.')