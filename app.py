import numpy as np
import uvicorn
import joblib
import mlflow
from pydantic import BaseModel
from fastapi import FastAPI

from common import MODEL_NAME, MODEL_STAGE


# Load latest production model from MLflow Registry
 

#model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.sklearn.load_model('./model')




app = FastAPI()

class diabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}

@app.post("/predict")
def predict_diabetes(data: diabetesInput):
    input_data = np.array([[data.pregnancies,data.glucose,data.blood_pressure,data.skin_thickness,
                            data.insulin,data.bmi,data.diabetes_pedigree_function,data.age]])
    
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
