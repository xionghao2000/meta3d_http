from dataclasses import dataclass


@dataclass
class Config3D:
    BUCKET_NAME: str
    REGION: str
    BUCKET_model_folder: str
