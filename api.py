import sys
sys.path.append('./src')

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from preprocess import preprocess_text
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # React app's address
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and label encoder
with open('models/nlp_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Define the input data model
class PatientInfo(BaseModel):
    age: int
    gender: str
    blood_type: str
    medical_condition: str
    medication: str

@app.post('/predict')
def predict(patient_info: PatientInfo):
    """Makes a prediction based on patient information."""
    # Create a pandas DataFrame from the input data
    data = {
        'Age': [patient_info.age],
        'Gender': [patient_info.gender],
        'Blood Type': [patient_info.blood_type],
        'text_features': [preprocess_text(patient_info.medical_condition + ' ' + patient_info.medication)]
    }
    df = pd.DataFrame(data)

    # Make a prediction
    prediction = model.predict(df)

    # Decode the prediction
    decoded_prediction = le.inverse_transform(prediction)

    return {"predicted_test_result": decoded_prediction[0]}
