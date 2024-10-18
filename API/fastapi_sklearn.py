import pandas as pd

from fastapi import FastAPI, UploadFile, File

import joblib

from pydantic import BaseModel


RN_STATE = 42

app = FastAPI()

model = joblib.load("bestmodel.pkl")

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
    df.rename(columns={
        'educational-num': 'educational_num',
        'marital-status': 'marital_status',
        'capital-gain': 'capital_gain',
        'capital-loss': 'capital_loss',
        'hours-per-week': 'hours_per_week',
        'native-country': 'native_country',
        'income_>50K': 'income_50K'
    }, inplace=True)
    # Define a mapping for categorical columns to integer values
    categorical_columns = [
        'workclass', 'education', 'marital_status', 'occupation',
        'relationship', 'race', 'gender', 'native_country'
    ]
    # Convert categorical columns to integers
    for column in categorical_columns:
        df[column] = df[column].astype('category').cat.codes
    # Convert remaining columns to appropriate integer types
    for column in df.columns:
        if column == 'age':
            df[column] = df[column].astype('uint8')
        else:
            df[column] = df[column].astype('uint32')
    return model_predict(df)

@app.post("/test")
async def test(payload: str):
    payload = payload + '!!'
    return {"test": payload}
