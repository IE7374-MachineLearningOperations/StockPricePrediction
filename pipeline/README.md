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

![DAG Run Summary](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_pipeline.png)

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
![Gantt Chart](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_gantt.jpeg)

#### Task Logs
Detailed logs for each task provide insights into the processing steps, including correlation matrix updates, data handling operations, and confirmation of successful execution steps. 

![Task Logs](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/v1.0/assets/airflow_logging.jpeg)

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