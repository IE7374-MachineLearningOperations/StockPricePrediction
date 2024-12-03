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
  ![Hyperparameter Sensitivity](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/results_linear_regression.png)

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
