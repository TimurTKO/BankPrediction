FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /opt/requirements.txt

RUN pip install --no-cache-dir streamlit

COPY stream.py .

CMD ["streamlit", "run", "stream.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
