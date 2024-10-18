FROM python:3.10-slim
USER root
WORKDIR /app
RUN pip install mlflow
EXPOSE 5000
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
