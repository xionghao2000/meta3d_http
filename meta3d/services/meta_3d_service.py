from regex import D
import torch
from tqdm.auto import tqdm
import uuid
import os
import boto3

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

from meta3d.common import config
from meta3d.services.s3_service import download_file

class MachineLearningService:
    def save(self, model, save_path):
        torch.save(model, save_path)
    def load(self, device, model_path):
        return torch.load(model_path, map_location=device)

class S3Service:
    def download_file(self, file_path, bucket, object_name):
        s3 = boto3.client('s3')
        s3.download_file(bucket, object_name, file_path)
    
    def check_exists(self, file_name: str):
        return os.path.exists(file_name)


class Meta3dService:
    def __init__(self, ml_service = MachineLearningService(), s3_service = S3Service()):
        self.ml_service = ml_service
        self.s3_service = s3_service
    
    def save_model(self, base_model, upsampler_model, model_path: str):
        '''
        save the model
        '''
        base_model_path = model_path + 'base_model.pt'
        unsample_model_path = model_path + 'upsample_model.pt'

        self.ml_service.save(base_model, base_model_path)
        self.ml_service.save(upsampler_model, unsample_model_path)
    
    def check_model(self, model_path: str):
        '''
        check if the model exists in the path
        '''
        base_model_path = model_path + 'base_model.pt'
        unsample_model_path = model_path + 'upsample_model.pt'

        if not os.path.exists(base_model_path):
            s3_b_model_path = config.BUCKET_model_folder + 'cup.ply'
            download_file(base_model_path, config.BUCKET_NAME, s3_b_model_path)
            
        if not os.path.exists(unsample_model_path):
            s3_un_model_path = config.BUCKET_model_folder + 'upsample_model.pt'
            print('no such file: ' + unsample_model_path)
            download_file(unsample_model_path, config.BUCKET_NAME, s3_un_model_path)


def create_model(device, base_name: str = 'base40M-textvec'):
    """
    create the model
    """
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()

    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()

    base_model.load_state_dict(load_checkpoint(base_name, device))

    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    return base_model, upsampler_model



def load_model(device, model_path: str):
    '''
    load the model
    '''
    base_model_path = model_path + 'base_model.pt'
    unsample_model_path = model_path + 'upsample_model.pt'

    base_model_loaded = torch.load(base_model_path, map_location=device)
    unsampler_diffusion_loaded = torch.load(unsample_model_path, map_location=device)
    return base_model_loaded, unsampler_diffusion_loaded

def create_diffusion(base_name: str):
    """
    create the diffusion
    """
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    return base_diffusion, upsampler_diffusion

def generate_3d_result(device, base_model, upsampler_model, base_diffusion, upsampler_diffusion, prompt: str):
    """
    generate the 3d model
    """
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    return pc

def save_model2ply(model ,ply_path: str):
    '''
    save the model to ply file
    '''
    filename = ply_path + str(uuid.uuid4()) + '.ply'

    with open(filename, 'wb') as f:
        model.write_ply(f)
    return filename