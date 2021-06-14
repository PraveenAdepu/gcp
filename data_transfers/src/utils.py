import pandas as pd
import io


def upload_local_file_to_gcp_storage_bucket(client, bucket, blob_name, data_file):
    for i in bucket:
        BUCKET = client.bucket(i)
        if BUCKET.exists():
            BUCKET = client.get_bucket(i)
            blob = BUCKET.blob(blob_name)
            blob.upload_from_filename(data_file)
            print("file upload completed")
        else:
            print("No bucket : {bucket}", bucket)


def gcp_csv_to_df(storage_client, bucket_name, blob_name, source_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_string()
    df = pd.read_csv(io.BytesIO(data))
    print(f"Pulled down file from bucket {bucket_name}, file name: {source_file_name}")
    return df


def df_to_gcp_csv(storage_client, df, bucket, blob_name, source_file_name):
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(df.to_csv(), "text/csv")
    print(f"DataFrame uploaded to bucket {bucket}, file name: {source_file_name}")
