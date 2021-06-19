import yaml as yaml
import pandas as pd
import numpy as np
import logging
# import pickle
from joblib import load
import os
from os.path import dirname, abspath
from sklearn.ensemble import RandomForestClassifier
import apache_beam as beam
from google.cloud import storage
from apache_beam.options.pipeline_options import (
    StandardOptions,
    GoogleCloudOptions,
    SetupOptions,
    PipelineOptions,
    WorkerOptions,
)

# feature engineering logic
dummy_dict = {"Yes": 1, "No": 0}
internet_dict = {"No": 0, "No internet service": 1, "Yes": 2}
train_yesno_cols = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]
inference_yesno_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
internet_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# preprocessing categorical features
def data_transformations(data, dummy_dict, internet_dict, yesno_cols, internet_cols):

    data[yesno_cols] = data[yesno_cols].apply(lambda x: x.map(dummy_dict))
    data[internet_cols] = data[internet_cols].apply(lambda x: x.map(internet_dict))

    # manual map
    data["gender"] = data["gender"].map({"Female": 0, "Male": 1})
    data["MultipleLines"] = data["MultipleLines"].map(
        {"No": 0, "No phone service": 1, "Yes": 2}
    )
    data["InternetService"] = data["InternetService"].map(
        {"DSL": 0, "Fiber optic": 1, "No": 2}
    )
    data["Contract"] = data["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    data["PaymentMethod"] = data["PaymentMethod"].map(
        {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }
    )
    return data
"""
# set project directory
project_directory = dirname(abspath("__file__"))

print("Processing : Loading configuration file")
config = yaml.safe_load(open(project_directory + "/config/config.yaml"))

print("Processing : Set Configuration parameters")
allservices_key = project_directory + config["parameters"]["all_services_account_key"]
"""
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = allservices_key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'C:\\Users\\61433\\Documents\\statscope\\GitHub\\gcp\\dataflow/keys/allservices/statscope-1c023b909ea7.json'

PROJECT = 'statscope'
JOB_NAME = 'batch-inference'
REGION = 'australia-southeast1'
BUCKET = 'dataflow-pipeline-batch-inference'
STAGING_LOCATION = 'gs://dataflow-pipeline-batch-inference/staging'
TEMP_LOCATION = 'gs://dataflow-pipeline-batch-inference/temp'
SERVICE_ACCOUNT_EMAIL = 'allservices@statscope.iam.gserviceaccount.com'
RUNNER = 'DataflowRunner'
SAVE_MAIN_SESSION = True
SETUP_FILE = './setup.py'
NUM_WORKERS = 4
AUTOSCALING_ALGORITHM = 'AUTOSCALING_ALGORITHM'
MODEL_PATH = 'model_artifacts/model_rf.joblib'
MODEL_ARTIFACT = 'model_rf.joblib'
PREDICTIONS_FILE = 'gs://dataflow-pipeline-batch-inference/predictions/'
WORKER_HARNESS_CONTAINER_IMAGE = 'asia.gcr.io/statscope/dataflow:latest'

"""
PROJECT = config["parameters"]["project"]
JOB_NAME = config["parameters"]["Dataflow_job_name"]
REGION = config["parameters"]["region"]
BUCKET = config["parameters"]["bucket"]
STAGING_LOCATION = (
    "gs://"
    + config["parameters"]["bucket"]
    + "/"
    + config["parameters"]["staging_folder"]
)
TEMP_LOCATION = (
    "gs://" + config["parameters"]["bucket"] + "/" + config["parameters"]["temp_folder"]
)
SERVICE_ACCOUNT_EMAIL = config["parameters"]["service_account_email"]
RUNNER = config["parameters"]["dataflow_runner"]
SAVE_MAIN_SESSION = bool(config["parameters"]["save_main_session"])
SETUP_FILE = config["parameters"]["setup_file"]
NUM_WORKERS = config["parameters"]["num_workers"]
AUTOSCALING_ALGORITHM = config["parameters"]["autoscaling_algorithm"]
MODEL_PATH = (
    config["parameters"]["model_artifacts_folder"]
    + "/"
    + config["parameters"]["blob_model_artifact"]
)
MODEL_ARTIFACT = config["parameters"]["blob_model_artifact"]
PREDICTIONS_FILE = (
    "gs://"
    + config["parameters"]["bucket"]
    + "/"
    + config["parameters"]["predictions_folder"]
    + "/"
)
"""

source_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# =============================================================================
# Build and run the pipeline
# =============================================================================
class FormatInput(beam.DoFn):
    def process(self, element):
        import pandas as pd

        """ Format the input to the desired shape"""
        df = pd.DataFrame([element], columns=source_columns)
        df = data_transformations(
            df, dummy_dict, internet_dict, inference_yesno_cols, internet_cols
        )
        df = df.fillna(0)
        output = df.to_dict("records")
        return output  # return a dict for easier comprehension


class PredictSklearn(beam.DoFn):
    def __init__(
        self, project=None, bucket_name=None, model_path=None, destination_name=None
    ):
        self._model = None
        self._project = project
        self._bucket_name = bucket_name
        self._model_path = model_path
        self._destination_name = destination_name

    def setup(self):
        from google.cloud import storage
        from joblib import load
        import pickle
        from sklearn.ensemble import RandomForestClassifier

        """Download sklearn model from GCS"""
        logging.info("Sklearn model initialisation {}".format(self._model_path))
        """
        download_blob(bucket_name=self._bucket_name, source_blob_name=self._model_path,
                      project=self._project, destination_file_name=self._destination_name)
        """
        # storage_key = "statscope-4fc1b774c9d0.json"
        # self.storage_client = storage.Client.from_service_account_json(storage_key)

        self.storage_client = storage.Client(self._project)
        self.bucket = self.storage_client.get_bucket(self._bucket_name)
        self.blob = self.bucket.blob(self._model_path)
        self.blob.download_to_filename(self._destination_name)
        # unpickle sklearn model
        self._model = load(self._destination_name)
        # with open(self._destination_name, "rb") as handle:
        #    self._model = pickle.load(handle)

    def process(self, element):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        """Inference from trained model artifact"""
        input_dat = {k: element[k] for k in element.keys() if k not in ["customerID"]}
        tmp = np.array(list(i for i in input_dat.values()))
        tmp = tmp.reshape(1, -1)
        element["prediction"] = self._model.predict_proba(tmp)[:, 1].item()
        # element["prediction"] = 0
        output = {
            k: element[k] for k in element.keys() if k in ["customerID", "prediction"]
        }
        output["customerID"] = str(output["customerID"])
        return [output]


def run(argv=None):
    pipeline_options = PipelineOptions(flags=argv)

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT
    google_cloud_options.job_name = JOB_NAME
    google_cloud_options.region = REGION
    google_cloud_options.staging_location = STAGING_LOCATION
    google_cloud_options.temp_location = TEMP_LOCATION
    google_cloud_options.service_account_email = SERVICE_ACCOUNT_EMAIL
    # pipeline_options.view_as(SetupOptions).experiment="beam_fn_api"
    pipeline_options.view_as(StandardOptions).runner = RUNNER
    pipeline_options.view_as(SetupOptions).save_main_session = SAVE_MAIN_SESSION
    pipeline_options.view_as(SetupOptions).setup_file = SETUP_FILE
    pipeline_options.view_as(WorkerOptions).num_workers = NUM_WORKERS
    pipeline_options.view_as(WorkerOptions).worker_harness_container_image = WORKER_HARNESS_CONTAINER_IMAGE
    pipeline_options.view_as(
        WorkerOptions
    ).autoscaling_algorithm = AUTOSCALING_ALGORITHM

    logging.info("Pipeline arguments: {}".format(pipeline_options))

    # table_schema = 'customerID: STRING, prediction: FLOAT'
    query = "SELECT * FROM `statscope.dataflow_pipeline_batch_inference.testingSet` LIMIT 10"
    bq_source = beam.io.BigQuerySource(query=query, use_standard_sql=True)
    p = beam.Pipeline(options=pipeline_options)
    (
        p
        | "Read data from BQ" >> beam.io.Read(bq_source)
        | "Preprocess data" >> beam.ParDo(FormatInput())
        | "predicting"
        >> beam.ParDo(
            PredictSklearn(
                project=PROJECT,
                bucket_name=BUCKET,
                model_path=MODEL_PATH,
                destination_name=MODEL_ARTIFACT,
            )
        )
        #     | "Write data to BQ" >> beam.io.WriteToBigQuery(table='prediction', dataset='Telco_Churn', project='statscope',
        #                                                     schema=table_schema,
        #                                                     # create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        #                                                     write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
        | "write to GCS"
        >> beam.io.WriteToText(PREDICTIONS_FILE, file_name_suffix=".csv")
    )

    result = p.run()
    result.wait_until_finish()


# log the output
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
