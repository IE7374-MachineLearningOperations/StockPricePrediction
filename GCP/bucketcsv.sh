#!/bin/bash

# Variables
LOCAL_REPO="/home/manormanore/Documents/Git_Hub/StockPricePrediction/"
BUCKET_NAME="stock_price_prediction_dataset"
BUCKET_PATH="Data"

gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"

TEMP_DIR=$(mktemp -d)

#files not excluded by .gitignore and matching the extensions (.csv and .png)
cd "$LOCAL_REPO"
git ls-files --cached --others --exclude-standard | grep -E "\.csv$|\.png$" | while read file; do
    mkdir -p "$TEMP_DIR/$(dirname "$file")"
    cp "$file" "$TEMP_DIR/$file"
done

# Sync the folder structure
gsutil -m rsync -r "$TEMP_DIR" "gs://$BUCKET_NAME/$BUCKET_PATH/"

rm -rf "$TEMP_DIR"

echo "Selected CSV and PNG files uploaded to gs://$BUCKET_NAME/$BUCKET_PATH/ excluding .gitignore entries"
