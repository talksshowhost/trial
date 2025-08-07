from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Загрузка пайплайна из файла
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
    
# Счетчик запросов
request_count = 0

class PredictionInput(BaseModel):
    Gender: str # Male or Female
    Age: int
    Education_Level: str # High School, Bachelor, Master, PhD
    Experience_Years: int
    Department: str # HR, Engineering, Marketing
    Job_Title: str # Analyst, Engineer, Manager
    Location: str # New York, San Francisco

@app.get('/stats')
def stats():
    return {'request_count': request_count}

@app.get('/health')
def health():
    return {'status': 'OK'}

@app.post('/predict_model')
def predict_model(input_data: PredictionInput):
    global request_count
    request_count+=1
    
    new_data = pd.DataFrame({
        'Age': [input_data.Age],
        'Gender': [input_data.Gender],
        'Department': [input_data.Department],
        'Job_Title': [input_data.Job_Title],
        'Experience_Years': [input_data.Experience_Years],
        'Education_Level': [input_data.Education_Level],
        'Location': [input_data.Location]
    })
    
    new_data['Gender'] = new_data['Gender'].map({'Male': 1, 'Female': 0})
    
    prediction = int(pipeline.predict(new_data))
    
    result = f'Worker salary is {prediction} dollars per year!'
    
    return {'prediction': result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000) 