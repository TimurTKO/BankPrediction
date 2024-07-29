FROM python:3.10
USER root
workdir /app
RUN pip install mlflow
ENTRYPOINT ["mlflow", "server", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
