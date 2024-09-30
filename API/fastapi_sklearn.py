import pandas as pd

from fastapi import FastAPI, UploadFile, File

import joblib

from pydantic import BaseModel


RN_STATE = 42

app = FastAPI()

model = joblib.load("../bestmodel.pkl")

def model_predict(df):
    y_pred = model.predict(df)
    return y_pred.tolist()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(file.file)
    return model_predict(df)


class DataFramePayload(BaseModel):
    json_str: str

@app.post("/receivedataframe")
async def receivedataframe(payload: DataFramePayload):
    # Convert the JSON string from the Pydantic model to a DataFrame
    df = pd.read_json(payload.json_str, orient='split')
    return model_predict(df)

@app.post("/test")
async def test(payload: str):
    payload = payload + '!!'
    return {"test": payload}
