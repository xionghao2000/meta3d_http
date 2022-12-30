import os

from dotenv import dotenv_values
_config = dotenv_values(".env")

BUCKET_NAME = _config.get("BUCKET_NAME","meta3d-bucket")
REGION = _config.get("REGION","ap-east-1231")