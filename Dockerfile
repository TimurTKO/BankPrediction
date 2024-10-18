FROM spark:3.5.1-scala2.12-java17-python3-r-ubuntu
USER root
WORKDIR /app
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=19888", "--no-browser", "--allow-root", "--NotebookApp.token='1234'", "--NotebookApp.password='1234'"]