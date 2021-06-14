# libraries
import pandas as pd
import yaml as yaml
from google.cloud import storage
from os.path import dirname, abspath

# utils
from utils import upload_local_file_to_gcp_storage_bucket, df_to_gcp_csv

# set project directory
project_directory = dirname(dirname(abspath("__file__")))

print("Processing : Loading configuration file")
config = yaml.safe_load(open(project_directory + "/config/config.yaml"))

print("Processing : Set Configuration parameters")
storage_key = project_directory + config["parameters"]["storage_service_account_key"]
data_file = project_directory + config["parameters"]["data_source"]
bucket = config["parameters"]["bucket_source"]
blob_name = config["parameters"]["blob_source"]

print("Processing : Set storage client")
storage_client = storage.Client.from_service_account_json(storage_key)

print("Processing : upload file")
upload_local_file_to_gcp_storage_bucket(storage_client, bucket, blob_name, data_file)

print("Processing : upload from pandas dataframe")
df = pd.read_csv(data_file)

df_to_gcp_csv(
    storage_client,
    df,
    bucket=bucket,
    blob_name=blob_name,
    source_file_name=blob_name,
)
