#!/bin/bash

# Set variables for GCP project and repository
PROJECT_ID="striped-graph-440017-d7"
LOCATION="us-east1"
REPOSITORY_NAME="mlopsgroup10"  # Your Artifact Registry repository name
KEY_FILE="/home/manormanore/Documents/Git_Hub/StockPricePrediction/GCP/striped-graph-440017-d7-c8fdb42bc8ba.json"

# Authenticate with GCP using the service account key
gcloud auth activate-service-account --key-file=$KEY_FILE

# Configure Docker to use the authenticated gcloud session
gcloud auth configure-docker "$LOCATION-docker.pkg.dev"

# Loop through each local Docker image
docker images --format "{{.Repository}}:{{.Tag}}" | while read IMAGE_NAME
do
    # Define the target tag for Artifact Registry
    TARGET_TAG="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}"

    # Tag and push the image
    docker tag "$IMAGE_NAME" "$TARGET_TAG"
    docker push "$TARGET_TAG"

    echo "Pushed $IMAGE_NAME to GCP Artifact Registry as $TARGET_TAG"
done

