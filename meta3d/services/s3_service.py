import logging
import os

import boto3
from botocore.exceptions import ClientError


def upload_file(file_name: str, region: str, bucketname: str, aws_access_key_id: str, aws_secret_access_key: str, object_name=None, endpoint_url=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3', region_name=region, aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url)
    try:
        response = s3_client.upload_file(file_name, bucketname, object_name)
    except ClientError as e:
        logging.error(e)
        return False


def get_url(object_name: str, bucketname: str, download_endpoint: str):
    '''
    get the url of the file
    '''
    url = download_endpoint + "/" + bucketname + "/" + object_name
    return url


def download_file(file_name: str, bucketname: str, region: str, aws_access_key_id: str, aws_secret_access_key: str, object_name=None, endpoint_url=None):
    '''
    download the file from s3
    '''
    s3 = boto3.client('s3', region_name=region, aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url)
    with open(file_name, 'wb') as f:
        s3.download_fileobj(bucketname, object_name, f)
