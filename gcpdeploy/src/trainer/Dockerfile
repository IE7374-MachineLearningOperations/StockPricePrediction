FROM python:3.9-slim
WORKDIR /app

ENV AIP_STORAGE_URI=gs://stock_price_prediction_dataset/model_checkpoints
ENV BUCKET_NAME=stock_price_prediction_dataset


# Copy current directory contents
COPY trainer/train.py /app/
COPY trainer/service_key.json /app/
COPY trainer/config.yaml /app/

# Install required dependencies
RUN pip install --no-cache-dir pandas numpy==1.21.2 google-cloud-storage scikit-learn joblib grpcio python-dotenv pyarrow wandb matplotlib

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "train"]


