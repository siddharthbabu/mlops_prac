# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

model = mlflow.sklearn.load_model("model")  # simpler if you don't want to set up the run ID

app = FastAPI()

class InputData(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Fare: float

@app.post("/predict")
def predict(data: InputData):
    x = np.array([[data.Pclass, data.Age, data.SibSp, data.Fare]])
    pred = model.predict(x)
    return {"survived": int(pred[0])}
