# Model Development Assignment Phase

This section explains the DAGs pipeline implemented using Apache Airflow for workflow orchestration. The approach focuses on stock market analysis, combining data from various sources including FRED (Federal Reserve Economic Data), Fama-French factors. The pipeline is organized with clear separation of concerns - data storage in the 'data' directory, processing scripts in 'src', and output artifacts for visualization. The source code includes essential ML preprocessing steps like handling missing values, feature engineering (including technical indicators and lagged features), dimensionality reduction through PCA, correlation analysis, hyperparameter tuning, and model sensitivity analysis.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Airflow Implementation](#airflow-implementation)
- [Pipeline Components](#pipeline-components)
- [Setup and Usage](#setup-and-usage)
- [Data Sources](#data-sources)

---

### Directory Structure

```
.
├── airflow
│   ├── artifacts                    
│   │   ├── correlation_matrix_after_removing_correlated_features.png
│   │   ├── Feature Importance for ElasticNet on Test Set.png
│   │   ├── Feature Importance for Lasso on Test Set.png
│   │   ├── Linear Regression - Hyperparameter Sensitivity: model__alpha.png
│   │   ├── Linear Regression - Hyperparameter Sensitivity: model__l1_ratio.png
│   │   ├── pca_components.png
│   │   └── yfinance_time_series.png
│   ├── dags                          # Contains Airflow DAGs
│   │   ├── airflow.py                # Main DAG for executing the pipeline
│   │   ├── data                      # Data sources 
│   │   │   ├── ADS_index.csv
│   │   │   ├── fama_french.csv
│   │   │   ├── FRED_Variables        
│   │   │   │   ├── AMERIBOR.csv
│   │   │   │   └── ...                # Other FRED data files
│   │   │   └── merged_original_dataset.csv
│   │   └── src                       # Scripts for various tasks
│   │       ├── convert_column_dtype.py
│   │       ├── correlation.py
│   │       ├── download_data.py
│   │       ├── feature_interactions.py
│   │       ├── handle_missing.py
│   │       ├── keep_latest_data.py
│   │       ├── lagged_features.py
│   │       ├── models
│   │       │   ├── linear_regression.py
│   │       │   ├── model_bias_detection.py
│   │       │   ├── model_sensitivity_analysis.py
│   │       └── pca.py
│   ├── docker-compose.yaml           # Docker configuration for running Airflow
│   ├── dvc.yaml                      # DVC configuration for data version control
│   ├── logs                          # Airflow logs
│   ├── plugins                      
│   └── working_data                  
└── pipelinetree.txt                  # Airflow working structure
```

### Airflow Implementation

The Airflow DAG (`Group10_DataPipeline_MLOps`) was successfully implemented and tested, with all tasks executing as expected. The following details summarize the pipeline's performance and execution.

#### DAG Run Summary
- **Total Runs**: 10
- **Last Run**: 2024-11-16 02:34:59 UTC
- **Run Status**: Success
- **Run Duration**: 00:03:37

#### Task Overview
The DAG consists of 19 tasks, each representing a step in the data processing, feature engineering, and model development pipeline. Key tasks include:
1. `download_data_task` - Downloads initial datasets from multiple financial data sources.
2. `convert_data_task` - Converts data types to ensure compatibility and efficiency.
3. `keep_latest_data_task` - Filters datasets to keep only the most recent data.
4. `remove_weekend_data_task` - Removes data points from weekends to ensure consistency in the time-series analysis.
5. `handle_missing_values_task` - Handles missing data using imputation techniques or removal where necessary.
6. `plot_time_series_task` - Visualizes the time series trends for better exploratory analysis.
7. `removing_correlated_variables_task` - Eliminates highly correlated variables to prevent redundancy.
8. `add_lagged_features_task` - Generates lagged features to capture temporal dependencies in the dataset.
9. `add_feature_interactions_task` - Creates new interaction features between existing variables for improved predictive power.
10. `add_technical_indicators_task` - Calculates financial technical indicators, such as moving averages and RSI.
11. `scaler_task` - Scales the dataset features to a consistent range to enhance model training stability.
12. `visualize_pca_components_task` - Performs PCA for dimensionality reduction and visualizes the key components.
13. `upload_blob_task` - Uploads processed data or model artifacts to cloud storage for later access.
14. `linear_regression_model_task` - Trains a linear regression model on the processed data.
15. `lstm_model_task` - Trains an LSTM model for time-series predictions.
16. `xgboost_model_task` - Trains an XGBoost model for predictive analysis.
17. `sensitivity_analysis_task` - Analyzes model sensitivity to understand feature importance and impact on predictions.
18. `detect_bias_task` - Detects any biases in the model by evaluating it across different data slices.
19. `send_email_task` - Sends notifications regarding the DAG's completion status to the stakeholders.

All tasks completed successfully with minimal execution time per task, indicating efficient pipeline performance.

#### Execution Graph and Gantt Chart
The **Execution Graph** confirms that tasks were executed sequentially and completed successfully (marked in green), showing no deferred, failed, or skipped tasks. The **Gantt Chart** illustrates the time taken by each task and confirms that the pipeline completed within the expected duration.

#### Execution Graph
![Execution Graph](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_graph.png)

#### Gantt Chart
![Gantt Chart](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_gantt.png)

#### Task Logs
Detailed logs for each task provide insights into the processing steps, including correlation matrix updates, data handling operations, and confirmation of successful execution steps. 

![Task Logs](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_pipeline.png)

#### Email Notifications 
**Anomaly Detection and Automated Alert**
Automated email notifications were configured to inform the team of task success or failure. As shown in the sample emails, each run completed with a success message confirming the full execution of the DAG tasks.

![Email Notifications](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/email_notification.jpeg)

#### Testing Summary
The pipeline scripts were validated with 46 unit tests using `pytest`. All tests passed with zero errors. These tests cover critical modules such as:
![Test Summary](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/test_functions.jpeg)
---

### Pipeline Components

1. **Data Extraction**:
   - `download_data.py`: Downloads datasets from various financial sources and loads them into the `data` directory.

2. **Data Preprocessing**:
   - `convert_column_dtype.py`: Converts data types for efficient processing.
   - `remove_weekend_data.py`: Filters out weekend data to maintain consistency in time-series analyses.
   - `handle_missing.py`: Handles missing data by imputation or removal.

3. **Feature Engineering**:
   - `correlation.py`: Generates a correlation matrix to identify highly correlated features.
   - `pca.py`: Performs Principal Component Analysis to reduce dimensionality and capture key components.
   - `lagged_features.py`: Creates lagged versions of features for time-series analysis.
   - `technical_indicators.py`: Calculates common technical indicators used in financial analyses.

4. **Data Scaling**:
   - `scaler.py`: Scales features to a standard range for better model performance.

5. **Analysis and Visualization**:
   - `plot_yfinance_time_series.py`: Plots time-series data.
   - `feature_interactions.py`: Generates interaction terms between features.

6. **Hyperparameter Tuning**:
   - Hyperparameter tuning was performed using grid search, random search, and Bayesian optimization. The best models were selected and saved as checkpoints in the `artifacts` directory. Visualizations for hyperparameter sensitivity, such as `Linear Regression - Hyperparameter Sensitivity: model__alpha.png` and `model__l1_ratio.png`, are also included in the `artifacts` directory.

7. **Model Sensitivity Analysis**:
   - Sensitivity analysis was conducted to understand how changes in inputs impact model performance. Techniques like **SHAP** (SHapley Additive exPlanations) were used for feature importance analysis. Feature importance graphs, such as `Feature Importance for ElasticNet on Test Set.png`, were saved to provide insights into key features driving model predictions.

### Setup and Usage

To set up and run the pipeline:

   > Refer to [Environment Setup](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#environment-setup) 

   > Refer to [Running Pipeline](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#running-the-pipeline)

### Data Sources

1. **ADS Index**: Tracks economic trends and business cycles.
2. **Fama-French Factors**: Provides historical data for financial research.
3. **FRED Variables**: Includes various economic indicators, such as AMERIBOR, NIKKEI 225, and VIX.
4. **YFinance**: Pulls historical stock data ('GOOGL') for financial time-series analysis.

---

# Model Development Pipeline Phase

## Contents
1. [Loading Data from the Data Pipeline](#1-loading-data-from-the-data-pipeline)
2. [Training and Selecting the Best Model](#2-training-and-selecting-the-best-model)
3. [Model Validation](#3-model-validation)
4. [Model Bias Detection (Using Slicing Techniques)](#4-model-bias-detection-using-slicing-techniques)
5. [Bias Detection and Mitigation Strategies](#5-bias-detection-and-mitigation-strategies)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Model Sensitivity Analysis](#7-model-sensitivity-analysis)
8. [Experiment Tracking and Results with Weights & Biases](#8-experiment-tracking-and-results-with-weights--biases)
---

### 1. Loading Data from the Data Pipeline
The model development process begins by loading data from a versioned dataset that has been processed through the data pipeline. The data should be consistent, cleaned, and preprocessed to maintain data quality.

- **Source**: Data is stored in Google Cloud Storage buckets.
- **Loading Method**: Data is retrieved using Google Cloud Storage APIs and loaded into memory.

**Bucket Tree Structure and Data Loading Image**:

```bash
.
├── Bucket: stock_price_prediction_dataset
│   ├── Codefiles                      # GCP resource and sync files
│   ├── models                         # Various ML model notebooks
│   ├── pipeline                       # Airflow DAGs, scripts, and tests
│   ├── src                            # Data preprocessing notebooks
│   └── DVC                            # Version control for datasets
```

![Data Pipeline](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_pipeline.png)

### 2. Training and Selecting the Best Model
Once the data is loaded, several machine learning models are trained to find the most suitable one based on performance metrics. 

- **Models Used**: Random Forest, ElasticNet, Ridge, Lasso.
- **Best Model Selection**: The model with the lowest Mean Squared Error (MSE) is selected as the best performing model.
- **Checkpointing**: The trained models are saved in the GCS bucket.

**Bucket Structure for Model Checkpoints**:

```bash
.
├── Bucket: stock_price_prediction_dataset
│   ├── model_checkpoints               # Checkpointed models
│       ├── ElasticNet.pkl
│       ├── LSTM.pkl
│       ├── Lasso.pkl
│       ├── Ridge.pkl
│       └── XGBoost.pkl
```

![Model Checkpoints](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/save_best_model_to_gcs.png)

### 3. Model Validation
Model validation is a crucial step to evaluate how well the selected model performs on unseen data. 

- **Metrics Used**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R^2 Score.
- **Validation Data**: A hold-out dataset is used for validating the model performance.

**Airflow Gantt Chart for Model Validation**:

![Airflow Gantt Chart](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_gantt.png)

### 4. Model Bias Detection (Using Slicing Techniques)
Bias detection ensures that the model behaves equitably across different subgroups of data.

- **Tools Used**: `Fairlearn` for data slicing and detecting model bias.
- **Purpose**: To evaluate model fairness across various demographic or sensitive features.

**Airflow Graph Depicting Bias Detection Task**:

![Airflow Graph](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_graph.png)

### 5. Bias Detection and Mitigation Strategies
If bias is detected in the model's predictions, mitigation strategies are employed to ensure fair performance.

- **Mitigation Techniques**:
  - **Re-sampling**: Resample the data using techniques like SMOTE to address class imbalances.
  - **Fairness Constraints**: Add fairness constraints during the training process, such as Demographic Parity.

**Bias Detection Logs**:

![Bias Detection Log](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/detect_bias_log.png)

### 6. Hyperparameter Tuning
Hyperparameter tuning is a critical step for optimizing the model’s performance.

- **Techniques Used**: Grid search, random search, and Bayesian optimization were utilized to determine the best set of hyperparameters.
- **Documented Search Space**: The search space and tuning process were meticulously documented for transparency and reproducibility in model optimization efforts.

**Bucket Structure for Hyperparameter Tuning**:

```bash
.
├── Bucket: stock_price_prediction_dataset
│   ├── model_checkpoints               # Models with different hyperparameter settings
```

![Artifacts Blob](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/artifacts_blob.png)

### 7. Model Sensitivity Analysis
Sensitivity analysis helps understand how changes in input features and hyperparameters affect model performance.

- **Feature Importance Analysis**: We used SHAP (SHapley Additive exPlanations) to evaluate feature importance and understand the impact of each feature on model predictions.
- **Hyperparameter Sensitivity Analysis**: The effects of varying hyperparameters on the model's output were explored, identifying those that had the most significant impact on overall performance.

**Sensitivity Analysis Visualizations**:

- **Feature Importance for ElasticNet**:

  ![ElasticNet Feature Importance](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/model_analysis_elasticNet.png)

- **Hyperparameter Sensitivity Analysis**:

  ![Hyperparameter Sensitivity](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/results_linear_regression .png)

### 8. Experiment Tracking and Results with Weights & Biases

In the model development process, we leveraged Weights & Biases (W&B) to meticulously track the progress of our experiments, providing a structured and comprehensive approach to model iteration and improvement.

- **Tracking Tools**: Weights & Biases served as the core platform for logging all experiments, tracking metrics, and visualizing the performance metrics of different model configurations. Using W&B allowed us to keep a consistent record of hyperparameters, model performance, and experimental conditions.

- **Logged Metrics**: We logged essential performance metrics, such as loss, accuracy, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Square Error (RMSE), for every model run. These metrics were crucial in comparing the performance of different models and understanding the impact of each experiment on the overall objective.

- **Experiment Visualization**: Weights & Biases provided insightful visualizations, which helped compare the performance across multiple models and experiments. This functionality was critical for making informed decisions about model optimization and improvement.

- **Run Comparison**: By comparing multiple experiments, we effectively identified the best performing model configuration. These comparisons were visually represented, allowing us to observe trends and key differences in performance across experiments.

**Experiment Tracking Visualizations**:

- **Overview of All Runs**:
  ![Overview of All Runs](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/overview_charts_all_runs.png)

- **Comparison of Different Runs**:
  ![Comparison of Runs](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/compare_different_runs.png)

- **Detailed View of a Single Run**:
  ![Detail of One Run](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/detail_one_run.png)

- **Main Dashboard Overview**:
  ![W&B Dashboard Overview](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/wandb_main_dashboard_overview_all_runs.png)


---

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