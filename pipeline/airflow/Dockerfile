# Dockerfile
FROM apache/airflow:2.6.0-python3.10

# Install required packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt