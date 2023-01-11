from dataclasses import dataclass


@dataclass
class Config3D:
    BUCKET_NAME: str
    REGION: str
    BUCKET_model_folder: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
