# DataPipeline Assignment Phase

This section explains dags pipeline implemented using Apache Airflow for workflow orchestration. The approach focus on stock market analysis, combining data from various sources including FRED (Federal Reserve Economic Data), Fama-French factors. The pipeline is organized with clear separation of concerns - data storage in the 'data' directory, processing scripts in 'src', and output artifacts for visualization. The source code includes essential ML preprocessing steps like handling missing values, feature engineering (including technical indicators and lagged features), dimensionality reduction through PCA, and correlation analysis. 

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
│   │   ├── pca_components.png
│   │   └── yfinance_time_series.png
│   ├── dags                          # Contains Airflow DAGs
│   │   ├── airflow.py                # Main DAG for executing the pipeline
│   │   ├── data                      # Data sources 
│   │   │   ├── ADS_index.csv
│   │   │   ├── fama_french.csv
│   │   │   ├── FRED_Variables        
│   │   │   └── merged_original_dataset.csv
│   │   └── src                       # Scripts for various tasks
│   │       ├── convert_column_dtype.py
│   │       ├── correlation.py
│   │       ├── download_data.py
│   │       ├── feature_interactions.py
│   │       ├── handle_missing.py
│   │       ├── keep_latest_data.py
│   │       ├── lagged_features.py
│   │       ├── pca.py
│   │       ├── plot_yfinance_time_series.py
│   │       ├── remove_weekend_data.py
│   │       ├── scaler.py
│   │       └── technical_indicators.py
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
- **Total Runs**: 1
- **Last Run**: 2024-11-05 05:34:59 UTC
- **Run Status**: Success
- **Run Duration**: 00:00:40

![DAG Run Summary](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/airflow_dags.jpeg)

#### Task Overview
The DAG consists of 13 tasks, each representing a step in data processing and feature engineering pipeline. Key tasks include:
1. `download_data_task` - Download initial datasets
2. `convert_data_task` - Convert data types as required
3. `handle_missing_values_task` - Handle missing data values
4. `add_feature_interactions_task` - Generate interaction features
5. `visualize_pca_components_task` - PCA visualization
6. `send_email_task` - Send success/failure notifications

All tasks completed successfully with minimal execution time per task, indicating efficient pipeline performance.

#### Execution Graph and Gantt Chart
The **Execution graph** confirms that tasks were executed sequentially and completed successfully (marked in green), showing no deferred, failed, or skipped tasks. The **Gantt chart** illustrates the time taken by each task and confirms that the pipeline completed within the expected duration.

#### Execution Graph
![Execution Graph](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/airflow_graph.jpeg)

#### Gantt Chart
![Gantt Chart](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/gantt.jpeg)

#### Task Logs
Detailed logs for each task provide insights into the processing steps, including correlation matrix updates, data handling operations, and confirmation of successful execution steps. 

![Task Logs](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/airflow_logging.jpeg)

#### Email Notifications 
 **Anomoly Detection and Automated Alert**
Automated email notifications were configured to inform the team of task success or failure. As shown in the sample emails, each run completed with a success message confirming the full execution of the DAG tasks.

![Email Notifications](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/email_notification.jpeg)

#### Testing Summary
The pipeline scripts were validated with 46 unit tests using `pytest`. All tests passed with zero errors. This main tests cover critical modules such as:
- `test_handle_missing.py`
- `test_feature_interactions.py`
- `test_plot_yfinance_time_series.py`

![Test Summary](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/blob/bf7526844544398e53ca528f30e883d1d87a493c/assets/test_functions.jpeg)

These tests ensure the stability and accuracy of data transformations, visualizations, and feature engineering processes.

---

### Pipeline Components

1. **Data Extraction**:
   - `download_data.py`: Downloads datasets from various financial sources and load and save from/to `data` directory.

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

### Setup and Usage

To set up and run the pipeline:

   > Refer to [Environment Setup](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#environment-setup) 

   > Refer to [Running Pipeline](https://github.com/IE7374-MachineLearningOperations/StockPricePrediction/tree/main?tab=readme-ov-file#running-the-pipeline)

### Data Sources

1. **ADS Index**: Tracks economic trends and business cycles.
2. **Fama-French Factors**: Provides historical data for financial research.
3. **FRED Variables**: Includes various economic indicators, such as AMERIBOR, NIKKEI 225, and VIX.
4. **YFinance**: Pulls historical stock data ('GOOGL') for financial time-series analysis. 
