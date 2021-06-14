import pandas as pd
import io
import yaml as yaml
from os.path import dirname, abspath
from google.cloud import bigquery
from google.cloud import storage

from src.utils import (create_bucket,create_folder,upload_local_file_to_gcp_storage_bucket,create_load_bigquery_table)

# set project directory
project_directory = dirname(abspath("__file__"))

print("Processing : Loading configuration file")
config = yaml.safe_load(open(project_directory + "/config/config.yaml"))

print("Processing : Set Configuration parameters")
allservices_key = project_directory + config["parameters"]["all_services_account_key"]
region = config["parameters"]["region"]
bucket = config["parameters"]["bucket"]
model_artifacts_folder = config["parameters"]["model_artifacts_folder"]
predictions_folder = config["parameters"]["predictions_folder"]
source_folder = config["parameters"]["source_folder"]
staging_folder = config["parameters"]["staging_folder"]
temp_folder = config["parameters"]["temp_folder"]
blob_source = config["parameters"]["blob_source"]
blob_source_data_file = project_directory + config["parameters"]["blob_source_data_file"]
blob_model_artifact = config["parameters"]["blob_model_artifact"]
blob_model_data_file = project_directory + config["parameters"]["blob_model_data_file"]
dataset = config["parameters"]["dataset"]
testingSet_table = config["parameters"]["testingSet_table"]


print("Processing : Set storage client")
storage_client = storage.Client.from_service_account_json(allservices_key)
bigquery_client = bigquery.Client.from_service_account_json(allservices_key)

print("Processing : Create bucket")
create_bucket(storage_client, bucket, region)

print("Processing : Create gcs folders")
create_folder(storage_client, bucket, model_artifacts_folder)
create_folder(storage_client, bucket, predictions_folder)
create_folder(storage_client, bucket, source_folder)
create_folder(storage_client, bucket, staging_folder)
create_folder(storage_client, bucket, temp_folder)

upload_local_file_to_gcp_storage_bucket(storage_client, bucket,folder=source_folder,blob_name=blob_source, data_file=blob_source_data_file)
upload_local_file_to_gcp_storage_bucket(storage_client, bucket,folder=model_artifacts_folder,blob_name=blob_model_artifact, data_file=blob_model_data_file)
         
gcs_source_file = "gs://"+bucket+"/"+source_folder+"/"+blob_source

# upload testingSet for inference into blob
create_load_bigquery_table(bigquery_client, dataset, testingSet_table, gcs_source_file)

# all done
# gcs bucket, folder, testingSet.csv and model artifact upload
# BigQuery table created and data loaded for pipeline inference
 