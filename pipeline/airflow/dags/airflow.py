from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.email import EmailOperator
from airflow import configuration as conf
import os
from dotenv import load_dotenv, dotenv_values
import yaml

from src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from src.convert_column_dtype import convert_type_of_columns
from src.keep_latest_data import keep_latest_data
from src.remove_weekend_data import remove_weekends
from src.handle_missing import fill_missing_values
from src.plot_yfinance_time_series import plot_yfinance_time_series
from src.correlation import removing_correlated_variables, plot_correlation_matrix
from src.lagged_features import add_lagged_features
from src.feature_interactions import add_feature_interactions
from src.technical_indicators import add_technical_indicators
from src.scaler import scaler
from src.pca import visualize_pca_components
from src.upload_blob import upload_blob

load_dotenv()
import os

# import wandb ## TODO
import sys

sys.path.append(os.path.abspath("."))
try:
    with open("dags/config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    config = {"WANDB_API_KEY": "----", "EMAIL_TO": "print.document.cb@gmail.com"}


os.environ["WANDB__SERVICE_WAIT"] = "300"
# wandb.login(key=config["WANDB_API_KEY"])  ## TODO


# Define function to notify failure or sucess via an email
def notify_success(context):
    success_email = EmailOperator(
        task_id="success_email",
        to=config["EMAIL_TO"],
        subject="Success Notification from Airflow",
        html_content="<p>The dag tasks succeeded.</p>",
        dag=context["dag"],
    )
    success_email.execute(context=context)


def notify_failure(context):
    failure_email = EmailOperator(
        task_id="failure_email",
        to=config["EMAIL_TO"],
        subject="Failure Notification from Airflow",
        html_content="<p>The dag tasks failed.</p>",
        dag=context["dag"],
    )
    failure_email.execute(context=context)


# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set("core", "enable_xcom_pickling", "True")
conf.set("core", "enable_parquet_xcom", "True")

# Define default arguments for your DAG
default_args = {
    "owner": "group_10",
    "start_date": datetime(2024, 10, 29),
    "retries": 0,  # Number of retries in case of task failure
    "retry_delay": timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'datapipeline' with the defined default arguments
dag = DAG(
    dag_id="Group10_DataPipeline_MLOps",
    default_args=default_args,
    description="Airflow DAG for the datapipeline",
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)


# Define the email task
send_email_task = EmailOperator(
    task_id="send_email_task",
    to=config["EMAIL_TO"],  # Email address of the recipient
    subject="Notification from Airflow",
    html_content="<p>This is a notification email sent from Airflow indicating that the dag was triggered</p>",
    dag=dag,
    on_failure_callback=notify_failure,
    on_success_callback=notify_success,
)

# Define PythonOperators for each function

# Task to download data from source, calls the 'download_data' Python function
download_data_task = PythonOperator(
    task_id="download_data_task",
    python_callable=merge_data,
    op_args=["GOOGL"],
    dag=dag,
)

# Task to convert column data types, calls the 'convert_type_of_columns' Python function
convert_type_data_task = PythonOperator(
    task_id="convert_data_task",
    python_callable=convert_type_of_columns,
    op_args=[download_data_task.output],
    dag=dag,
)


# Task to keep the latest data, calls the 'keep_latest_data' Python function
keep_latest_data_task = PythonOperator(
    task_id="keep_latest_data_task",
    python_callable=keep_latest_data,
    op_args=[convert_type_data_task.output, 10],
    dag=dag,
)

# Task to remove weekend data, calls the 'remove_weekends' Python function
remove_weekend_data_task = PythonOperator(
    task_id="remove_weekend_data_task",
    python_callable=remove_weekends,
    op_args=[keep_latest_data_task.output],
    dag=dag,
)

# Task to handle missing values, calls the 'fill_missing_values' Python function
handle_missing_values_task = PythonOperator(
    task_id="handle_missing_values_task",
    python_callable=fill_missing_values,
    op_args=[remove_weekend_data_task.output],
    dag=dag,
)

# Task to plot the time series data, calls the 'plot_yfinance_time_series' Python function
plot_time_series_task = PythonOperator(
    task_id="plot_time_series_task",
    python_callable=plot_yfinance_time_series,
    op_args=[handle_missing_values_task.output],
    dag=dag,
)

# Task to remove correlated variables task, calls the 'removing_correlated_variables' Python function
removing_correlated_variables_task = PythonOperator(
    task_id="removing_correlated_variables_task",
    python_callable=removing_correlated_variables,
    op_args=[handle_missing_values_task.output],
    dag=dag,
)

# Task to add lagged features, calls the 'add_lagged_features' Python function
add_lagged_features_task = PythonOperator(
    task_id="add_lagged_features_task",
    python_callable=add_lagged_features,
    op_args=[removing_correlated_variables_task.output],
    dag=dag,
)

# Task to add feature interactions, calls the 'add_feature_interactions' Python function
add_feature_interactions_task = PythonOperator(
    task_id="add_feature_interactions_task",
    python_callable=add_feature_interactions,
    op_args=[add_lagged_features_task.output],
    dag=dag,
)

# Task to add technical indicators, calls the 'add_technical_indicators' Python function
add_technical_indicators_task = PythonOperator(
    task_id="add_technical_indicators_task",
    python_callable=add_technical_indicators,
    op_args=[add_feature_interactions_task.output],
    dag=dag,
)

# Task to scale the data, calls the 'scaler' Python function
scaler_task = PythonOperator(
    task_id="scaler_task",
    python_callable=scaler,
    op_args=[add_technical_indicators_task.output],
    dag=dag,
)


visualize_pca_components_task = PythonOperator(
    task_id="visualize_pca_components_task",
    python_callable=visualize_pca_components,
    op_args=[scaler_task.output],
    dag=dag,
)

# Task to upload the data to Google Cloud Storage
upload_blob_task = PythonOperator(
    task_id="upload_blob_task",
    python_callable=upload_blob,
    op_args=[scaler_task.output],
    dag=dag,
)


# Set task dependencies
(
    download_data_task
    >> convert_type_data_task
    >> keep_latest_data_task
    >> remove_weekend_data_task
    >> handle_missing_values_task
    >> plot_time_series_task
    >> removing_correlated_variables_task
    >> add_lagged_features_task
    >> add_feature_interactions_task
    >> add_technical_indicators_task
    >> scaler_task
    >> visualize_pca_components_task
    >> upload_blob_task
    >> send_email_task
)

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    print("DAG is being run directly...")
    dag.cli()
