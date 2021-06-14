# libraries
import pandas as pd
import yaml as yaml
from google.cloud import storage
from os.path import dirname, abspath
import io

# utils, all functions import here
from utils import gcp_csv_to_df

project_directory = dirname(dirname(abspath("__file__")))

print("Processing : Loading configuration file")
config = yaml.safe_load(open(project_directory + "/config/config.yaml"))

print("Processing : Set configuration parameters")
storage_key = project_directory + config["parameters"]["storage_service_account_key"]
bucket = config["parameters"]["bucket_features"][0]  # string value from list
blob_name = config["parameters"]["blob_features"]

print("Processing : Set storage client")
storage_client = storage.Client.from_service_account_json(storage_key)

print("Processing : download file")
df = gcp_csv_to_df(storage_client, bucket, blob_name, source_file_name=blob_name)

df.shape
