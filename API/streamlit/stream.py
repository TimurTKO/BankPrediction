import streamlit as st
import pandas as pd
import requests
import os

#st.write("Переменные окружения:", os.environ)
st.title("Предсказание на основе модели")
st.write("Загрузите файл с данными для предсказания")
st.write(os.environ.get('FASTAPI_URL'))

uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

click = st.sidebar.button("Predict")
if click:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        json_str = df.to_json(orient='split')
        payload = {
            "json_str": json_str
        }

        FASTAPI_URL = os.getenv('FASTAPI_URL')
        
        # Проверка, установлен ли FASTAPI_URL
        if not FASTAPI_URL:
            st.error("URL сервиса FastAPI не найден.")
        else:
            url = f"{FASTAPI_URL}/receivedataframe"
            response = requests.post(url, json=payload)
            
            # Обработка ответа только если запрос успешен
            if response.status_code == 200:
                predictions = response.json()
                df_predictions = pd.DataFrame(data={"predictions": predictions})
                csv = df_predictions.to_csv(index=False)

                st.download_button(
                    label="Скачать результаты",
                    data=csv,
                    file_name="results.csv",
                    mime="text/csv"
                )
            else:
                st.error("Ошибка при запросе к сервису FastAPI. Проверьте URL и доступность сервиса.")
