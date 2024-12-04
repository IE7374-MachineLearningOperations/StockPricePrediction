FROM python:3.9-slim
WORKDIR /app

ENV AIP_STORAGE_URI=gs://stock_price_prediction_dataset/model_checkpoints
ENV BUCKET_NAME=stock_price_prediction_dataset


# Copy current directory contents
COPY trainer/train.py /app/

# Install required dependencies
RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio python-dotenv pyarrow 

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "train"]


