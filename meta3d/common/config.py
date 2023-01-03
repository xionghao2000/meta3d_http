from dotenv import dotenv_values
import os

_config = dotenv_values(".env")

BUCKET_NAME = os.getenv("BUCKET_NAME", "meta3d-bucket")
BUCKET_model_folder = os.getenv("BUCKET_model_folder", "meta3d-bucket")
REGION = os.getenv("REGION", "ap-east-1231")
