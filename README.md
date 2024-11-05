# Project Title: Stock-Price-Prediction

## Description
The primary goal of this project is to improve stock price forecasting and prediction by creating an effective Machine Learning Operations (MLOps) pipeline. Utilising sophisticated financial modelling methods like the GARCH and Kalman filters, the pipeline seeks to maximise model performance and increase the precision of stock volatility forecasts. In order to provide scalability and flexibility to new data and market trends, the project builds an automated pipeline that facilitates continuous integration and delivery of machine learning models. The system is able to react dynamically to shifting market conditions because of the smooth integration of fresh data made possible by this scalable MLOps architecture. In the end, this helps academics, data scientists, and financial analysts who want to use MLOps to make better investment decisions by producing forecasts that are more accurate.

---
## DataPipeline Assignment Phase

**Key Features:**
- Airflow DAGs for data processing pipeline.
- Correlation analysis and PCA for data insights and dimensionality reduction.
- Visualization of financial time series data.
---
## Table of Contents
1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Running the Pipeline](#running-the-pipeline)
4. [Test Functions](#test-functions)
5. [Reproducibility and Data Versioning](#reproducibility-and-data-versioning)
---

## Project Structure
```bash
.
├── assets/                 # Visualization outputs and project related figures
├── data/                   # Data storage (like datasets)
├── pipeline/               # Pipeline components
│   └── airflow/
│       ├── dags/          # Airflow DAGs
│       │   ├── data/      # Datasets
│       │   └── src/       # Pipeline scripts
│       ├── tests/         # Pytests
│       ├── logs/          # Airflow Execution logs
│       ├── dvc.yaml       # DVC pipeline definition
│       ├── README.md      # Documentation
│       └── docker-compose.yaml  # Airflow configuration
├── src/                    # Core source code
├── tests/                  # Pytests
├── .gitignore             # Git ignore patterns
├── .dvcignore             # DVC ignore patterns
├── README.md              # Project documentation
└── current.txt            # Repo tree structure
```

**Key Components:**
- `data/`: Contains raw and processed datasets 
- `pipeline/airflow/dags/`: DAGs for data processing.
- `pipeline/airflow/src/`: Core scripts for data transformation, feature engineering, and plotting.
- `tests/`: pytests for verifying code functionality.
- `assets/`: Contains generated plots for data analysis and visualization.
---

## Environment Setup

### Prerequisites

To set up and run this project, ensure the following are installed:

- **Python** (3.8 or later)
- **Docker** (for running Apache Airflow)
- **DVC** (for data version control)
- **Google Cloud SDK** (we are deploying on GCP)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/IE7374-MachineLearningOperations/StockPricePrediction.git
   cd Stock-Price-Prediction
   ```

2. **Install Python Dependencies**
   Install all required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC**
   Set up DVC to manage large data files by pulling the tracked data:
   ```bash
   dvc pull
   ```
---

## Running the Pipeline

To execute the data pipeline, follow these steps:

1. **Start Airflow Services**
   Run Docker Compose to start services of the Airflow web server, scheduler:
   ```bash
   cd pipeline/airflow/
   docker-compose up
   ```

2. **Access Airflow UI**
   Open `http://localhost:8080` in your browser. Log into the Airflow UI and enable the DAG

3. **Trigger the DAG**
   Trigger the DAG manually to start processing. The pipeline will:
   - Ingest raw data and preprocess it.
   - Perform correlation analysis to identify redundant features.
   - Execute PCA to reduce dimensionality.
   - Generate visualizations, such as time series plots and correlation matrices.

4. **Check Outputs**
   Once completed, check the output files and images in the `assets/` folder.

or 

```sh
# Step 1: Activate virtual environment: 
cd airflow_env/ # (go to Airflow environment and open in terminal)
source bin/activate

# Step 2: Install Airflow (not required if done before)
pip install apache-airflow

# Step 3: Initialize Airflow database (not required if done before)
airflow db init

# Step 4: Start Airflow web server and airflow scheduler
airflow webserver -p 8080 & airflow scheduler

# Step 5: Access Airflow UI in your default browser
# http://localhost:8080

# Step 6: Deactivate virtual environment (after work completion)
deactivate
```

---
## Test Functions
   Run all tests in the `tests` directory
   ```bash
   pytest tests/
   ```
---
## Reproducibility and Data Versioning

We used **DVC (Data Version Control)** for reproducibility and files management.

### DVC Setup
1. **Initialize DVC** (already initialized in the repository):
   ```bash
   dvc init
   ```

2. **Pull Data Files**
   Pull the DVC-tracked data files to ensure all required datasets are available:
   ```bash
   dvc pull
   ```

3. **Data Versioning**
   Data files are generated with `.dvc` files in the repository

4. **Tracking New Data**
   If new files are added, track them. Example:
   ```bash
   dvc add <file-path>
   dvc push
   ```
---
