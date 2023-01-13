import os

import boto3
import point_e.models.configs
import torch
from meta3d.common.config_3d import Config3D
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.diffusion.sampler import PointCloudSampler as Sampler
from point_e.models.configs import MODEL_CONFIGS
from point_e.models.download import load_checkpoint
from point_e.util.plotting import plot_point_cloud

from tqdm.auto import tqdm


class PointCloudSampler(Sampler):
    def __init__(self, device, models: list, diffusions: list, num_points: list, aux_channels: list,
                 guidance_scale: list,
                 model_kwargs_key_filter: tuple):
        super().__init__(
            device=device,
            models=models,
            diffusions=diffusions,
            num_points=num_points,
            aux_channels=aux_channels,
            guidance_scale=guidance_scale,
            model_kwargs_key_filter=model_kwargs_key_filter,
        )


class MachineLearningService:
    def save(self, model, save_path):
        torch.save(model, save_path)

    def load(self, model_path, map_location):
        return torch.load(model_path, map_location=map_location)


class S3Service:
    def download_file(self, file_path, bucket, object_name, region, aws_access_key_id, aws_secret_access_key, endpoint_url=None):
        s3 = boto3.client('s3', region_name=region, aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key, endpoint_url=endpoint_url)
        s3.download_file(bucket, object_name, file_path)

    def check_exists(self, file_name: str):
        return os.path.exists(file_name)


class PointEService:
    class Model:
        def eval(self):
            pass

        def load_state_dict(self, val):
            pass

        def write_ply(self, val):
            raise NotImplementedError

    def model_from_config(self, model_config, device) -> Model:
        return point_e.models.configs.model_from_config(model_config, device)

    def load_checkpoint(self, base_name, device):
        return point_e.models.download.load_checkpoint(base_name, device)

    def diffusion_from_config(self, diffusion_config):
        return point_e.diffusion.configs.diffusion_from_config(diffusion_config)
    
    def plot_point_cloud(self, model, color, grid_size, fixed_bounds):
        return point_e.util.plotting.plot_point_cloud(model, color, grid_size, fixed_bounds)


class Meta3dService:
    def __init__(self, config: Config3D, pointe_service=PointEService(), s3_service=S3Service(),
                 ml_service=MachineLearningService()):
        self.pointe_service = pointe_service
        self.s3_service = s3_service
        self.ml_service = ml_service
        self.config = config

    def save_model(self, base_model, upsampler_model, model_path: str):
        '''
        save the model
        '''
        base_model_path = model_path + 'base_model.pt'
        unsample_model_path = model_path + 'upsample_model.pt'

        self.ml_service.save(base_model, base_model_path)
        self.ml_service.save(upsampler_model, unsample_model_path)

    def load_model(self, device, model_path: str):
        '''
        load the model
        '''
        base_model_path = model_path + 'base_model.pt'
        unsample_model_path = model_path + 'upsample_model.pt'

        base_model_loaded = self.ml_service.load(
            model_path=base_model_path, map_location=device)
        unsampler_model_loaded = self.ml_service.load(
            model_path=unsample_model_path, map_location=device)
        return base_model_loaded, unsampler_model_loaded

    def check_model(self, model_path: str):
        '''
        check if the model exists in the path
        '''
        base_model_path = model_path + 'base_model.pt'
        unsample_model_path = model_path + 'upsample_model.pt'

        if not self.s3_service.check_exists(base_model_path):
            s3_b_model_path = self.config.BUCKET_model_folder + 'base_model.pt'
            self.s3_service.download_file(file_path=base_model_path, bucket=self.config.BUCKET_NAME, region=self.config.REGION,
                                          object_name=s3_b_model_path, aws_access_key_id=self.config.AWS_ACCESS_KEY_ID, aws_secret_access_key=self.config.AWS_SECRET_ACCESS_KEY, endpoint_url=self.config.ENDPOINT_URL)

        if not self.s3_service.check_exists(unsample_model_path):
            s3_un_model_path = self.config.BUCKET_model_folder + 'upsample_model.pt'
            self.s3_service.download_file(file_path=unsample_model_path, bucket=self.config.BUCKET_NAME, region=self.config.REGION,
                                          object_name=s3_un_model_path, aws_access_key_id=self.config.AWS_ACCESS_KEY_ID, aws_secret_access_key=self.config.AWS_SECRET_ACCESS_KEY, endpoint_url=self.config.ENDPOINT_URL)

    def create_model(self, device, base_name: str = 'base40M-textvec'):
        """
        create the model
        """
        base_model = self.pointe_service.model_from_config(
            model_config=MODEL_CONFIGS[base_name], device=device)
        base_model.eval()

        upsampler_model = self.pointe_service.model_from_config(
            model_config=MODEL_CONFIGS['upsample'], device=device)
        upsampler_model.eval()

        base_model.load_state_dict(
            self.pointe_service.load_checkpoint(base_name, device))

        upsampler_model.load_state_dict(
            self.pointe_service.load_checkpoint('upsample', device))

        return base_model, upsampler_model

    def create_diffusion(self, base_name: str = 'base40M-textvec'):
        """
        create the diffusion
        """
        base_diffusion = self.pointe_service.diffusion_from_config(
            diffusion_config=DIFFUSION_CONFIGS.get(base_name, ""))
        upsampler_diffusion = self.pointe_service.diffusion_from_config(
            diffusion_config=DIFFUSION_CONFIGS.get('upsample', ""))
        return base_diffusion, upsampler_diffusion

    def generate_3d_result(self, sampler: PointCloudSampler, prompt: str):
        """
        generate the 3d model
        """
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]
        return pc

    def save_model2ply(self, model, ply_path: str):
        '''
        save the model to ply file
        '''
        filename = ply_path + 'result' + '.ply'

        with open(filename, 'wb') as f:
            model.write_ply(f)
        return filename

    def generate_model_image(self, model, grid_size):
        '''
        save the model to image
        '''
        imageContent = self.pointe_service.plot_point_cloud(model,grid_size=3, color=True, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
        return imageContent

    def convert_image_to_bytes(self, imageContent):
        import io
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        imageContent.savefig(buf, format='png')
        buf.seek(0)
        return buf


if __name__ == '__main__':
    prompt = input("Please enter the prompt: ")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    service = Meta3dService()

    check_model = service.check_model(
        model_path='D:\\RJdeck\\Metatopia\\meta3d\\meta3d\\models\\')
    base_model, upsampler_model = service.load_model(device=device,
                                                     model_path='D:\\RJdeck\\Metatopia\\meta3d\\meta3d\\models\\')
    base_diffusion, upsampler_diffusion = service.create_diffusion()

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),
    )
    pc = service.generate_3d_result(sampler, prompt=prompt)
