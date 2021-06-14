# util functions
from google.cloud import bigquery


def create_bucket(storage_client, bucket, region):
    bucket = storage_client.bucket(bucket)
    bucket.location = region
    bucket.create()


def create_folder(storage_client, bucket, folder):
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(folder + "/")
    blob.upload_from_string("")


def upload_local_file_to_gcp_storage_bucket(
    storage_client, bucket, folder, blob_name, data_file
):
    BUCKET = storage_client.get_bucket(bucket)
    blob = BUCKET.blob(folder + "/" + blob_name)
    blob.upload_from_filename(data_file)


def create_load_bigquery_table(bigquery_client, dataset, table, gcs_source_file):

    jobConfig = bigquery.LoadJobConfig()
    jobConfig.skip_leading_rows = 1
    jobConfig.source_format = bigquery.SourceFormat.CSV
    jobConfig.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    jobConfig.autodetect = True

    tableRef = bigquery_client.dataset(dataset).table(table)
    bigqueryJob = bigquery_client.load_table_from_uri(
        gcs_source_file, tableRef, job_config=jobConfig
    )
    bigqueryJob.result()
