FROM python:3.10
USER root
workdir /app
COPY requirements.txt /app
COPY fastapi_sklearn.py /app
RUN pip install -r requirements.txt
uvicorn fastapi_sklearn:app --host 127.0.0.1 --port 8047 --reload


