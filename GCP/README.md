# Model Development Phase

## CI/CD Pipeline Documentation and Structure Overview

The project folder structure is as follows:
- **Cloud Build Trigger Setup** - GitHub repository is connected with Google Cloud Build.
- **Model Checkpoints** - Stored models (`ElasticNet.pkl`, `LSTM.pkl`, etc.) are saved in the GCP bucket for checkpoints.
- **Airflow Pipeline** - Automated data extraction, preprocessing, and model training tasks.
- **VM Instances and Cloud Run** - Utilized for hosting and triggering various components of the pipeline.

## Setup and Tools Used
The following GCP components are used in this pipeline:

1. **Google Cloud Storage (GCS)**: Stores datasets, models, and artifacts. Bucket name used: `stock_price_prediction_dataset`, which contains the following directories:
  - `Codefiles/`: Contains source code files for the pipeline.
  - `Data/`: Stores input datasets used for training and validation.
  - `DVC/`: Version control for datasets and other artifacts.
  - `gs:/`: GCP-specific files and configurations.
  - `model_checkpoints/`: Stores different model checkpoint files such as `ElasticNet.pkl`, `LSTM.pkl`, `Lasso.pkl`, etc.
2. **Google Cloud Composer (Airflow)**: Orchestrates ETL workflows and model training. DAG folder used: `gs://us-central1-mlopscom10-0658b5dc-bucket/dags`.
3. **Google Cloud Build**: Triggers on commits to the main or test branch in the GitHub repository to start the CI/CD pipeline.
4. **Google Cloud Run**: Used for serverless execution and model deployment. Cloud function name: `mlops10trigger`.
5. **Google Artifact Registry**: Stores versioned models.
6. **GitHub Actions**: Handles CI/CD and rollback for the GitHub repository.

## GCP Buckets Overview


```
.
├── Bucket: buc-logs                   # Stores log files for auditing
├── Bucket: cloud-ai-platform          # Artifacts from AI platform
├── Bucket: gcf-v2-sources             # Source data for cloud triggers
│   ├── cloud-trigger-data
│   └── mlops10trigger
├── Bucket: gcf-v2-uploads             # Cloud Function uploads
├── Bucket: stock_price_prediction_dataset
│   ├── Codefiles                      # GCP resource and sync files
│   ├── models                         # Various ML model notebooks
│   ├── pipeline                       # Airflow DAGs, scripts, and tests
│   ├── src                            # Data preprocessing notebooks
│   └── DVC                            # Version control for datasets
└── Bucket: us-central1-mlopscom10-bucket
    ├── dags                           # Airflow orchestration files
    └── src                            # Supporting Python scripts for DAGs
```

## Setting Up Cloud Build Trigger
Cloud Build is configured to trigger the build pipeline automatically when a new commit is pushed to the main branch in GitHub. The build trigger is named `StockMlopps` and is configured to monitor the repository `IE7374-MachineLearningOperations/StockPricePrediction`.

![Cloud Build Trigger Setup](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/github_trigger.png)

## Airflow Environment Details
The Airflow instance in **Google Cloud Composer** handles tasks like downloading data, preprocessing, model training, and uploading results. The screenshot below shows the details of the Airflow environment:

![Airflow Environment](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_gcp.png)

### Airflow DAG Statistics
Below are the statistics for the successful execution of DAG runs, showing a stable orchestration:

![Airflow DAG Statistics](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/dags_run.png)

## Google Cloud Run for Model Deployment
The model is deployed using **Google Cloud Run** after a successful DAG run. Cloud Run offers serverless functionality to manage and deploy trained models effectively. Below is a view of the VM instance utilized for other processes in this setup:

![VM Instances in GCP](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/VM_instance.png)

## Rollback Mechanism
The CI/CD pipeline also includes a **rollback mechanism** for both model versions and deployments.

### Rolling Back Model Deployment
A **Cloud Run trigger** (`mlops10trigger`) can be configured to roll back to a previous stable version in case any issue arises with the latest deployment.

![Cloud Run Trigger for Rollback](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/mlops10trigger.png)

## Model Registry in Artifact Storage

The trained models are automatically pushed to **GCP Artifact Registry** to store and manage different versions of the models as they get updated. The artifact repository used is `us-east1-docker.pkg.dev/striped-graph-440017-d7/gcf-artifacts`, which contains Docker images used for deployment, such as `striped-graph-440017-d7_us-east1_mlops10trigger`.

![Artifact Registry](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/gcp-artifcats.png)

### Model Checkpoints Saved in GCP Bucket
All the model checkpoints, including `ElasticNet.pkl`, `LSTM.pkl`, `Lasso.pkl`, etc., are saved in the GCS bucket for version tracking and recovery.

![Model Checkpoints in GCP Bucket](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/model_checkpoints.png)

## GitHub Actions for CI/CD
**GitHub Actions** is used for CI/CD automation of the repository. It automatically builds and deploys updates after successful commits.

### Cloud Build Trigger and Artifact Publishing
A detailed screenshot of the CI/CD trigger setup and successful execution logs from GitHub is shown below:

![GitHub Actions - CI/CD Pipeline](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/mlops10trigger.png)