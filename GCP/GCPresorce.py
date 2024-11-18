from google.cloud import compute_v1
from google.cloud import storage
from googleapiclient import discovery
from google.oauth2 import service_account

project_id = 'striped-graph-440017-d7'
zone = 'us-east1-a' 

def list_gcp_compute_instances(project_id, zone):
    """Lists all VM instances in a specified project and zone."""
    compute_client = compute_v1.InstancesClient()
    instances = compute_client.list(project=project_id, zone=zone)
    print("VM Instances:")
    for instance in instances:
        print(f" - Name: {instance.name}, Status: {instance.status}, Zone: {instance.zone}")

def list_gcp_storage_buckets(project_id):
    """Lists all Storage Buckets in a specified project."""
    storage_client = storage.Client(project=project_id)
    buckets = storage_client.list_buckets()
    print("Storage Buckets:")
    for bucket in buckets:
        print(f" - Name: {bucket.name}, Location: {bucket.location}")

def list_gcp_sql_instances(project_id, credentials_path):
    """Lists all Cloud SQL instances in a specified project using the REST API."""
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    sql_service = discovery.build('sqladmin', 'v1', credentials=credentials)
    request = sql_service.instances().list(project=project_id)
    response = request.execute()
    print("SQL Instances:")
    if 'items' in response:
        for instance in response['items']:
            print(f" - Name: {instance['name']}, DB Version: {instance['databaseVersion']}, Region: {instance['region']}")
    else:
        print("No Cloud SQL instances found.")

# Provide the path to your service account key
credentials_path = '/home/manormanore/Documents/Git_Hub/StockPricePrediction/striped-graph-440017-d7-79f99f8253bc.json'

# Run the functions
list_gcp_compute_instances(project_id, zone)
list_gcp_storage_buckets(project_id)
list_gcp_sql_instances(project_id, credentials_path)

