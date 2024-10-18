import streamlit as st
import pandas as pd
import requests



st.title("Предсказание на основе модели")
st.write("Загрузите файл с данными для предсказания")

uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

click = st.sidebar.button("Predict")
if click:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        json_str = df.to_json(orient='split')
        payload = {
            "json_str": json_str
        }


        url = "http://fastapi:8000/receivedataframe"
        
        response = requests.post(url, json=payload)

        predictions = response.json()

        df_predictions = pd.DataFrame(data={"predictions": predictions})


        csv = df.to_csv(index=False)

        st.download_button(
            label="Скачать результаты", 
            data=csv,  # передаем строку CSV, а не DataFrame
            file_name="results.csv", 
            mime="text/csv"
        )


