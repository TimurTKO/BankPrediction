FROM python:3.10-slim
USER root
workdir /abc
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY ./bestmodel.pkl .
COPY fastapi_sklearn.py .
ENTRYPOINT ["uvicorn", "fastapi_sklearn:app", "--host", "0.0.0.0", "--port", "8000"]



