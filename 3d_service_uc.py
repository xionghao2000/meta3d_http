import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

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

def save_model(base_model, upsampler_model, base_model_path: str, unsample_model_path: str):
    '''
    save the model
    '''
    torch.save(base_model, base_model_path)
    torch.save(upsampler_model, unsample_model_path)

def load_model(device, base_model_path: str, unsample_model_path: str):
    '''
    load the model
    '''
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

def generate_3dmodel(device, base_model, upsampler_model, base_diffusion, upsampler_diffusion, prompt: str):
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

def save_model2ply(model ,filename: str):
        '''
        save the model to ply file
        '''
        with open(filename, 'wb') as f:
            model.write_ply(f)

base_name = 'base40M-textvec'
base_model_path = 'D:\\RJdeck\\Metatopia\\meta3d\\base_model.pt'
unsample_model_path = 'D:\\RJdeck\\Metatopia\\meta3d\\upsample_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model_load, upsampler_model_load = load_model(device,base_model_path ,unsample_model_path )
base_diffusion, upsampler_diffusion = create_diffusion(base_name)
pc = generate_3dmodel(device, base_model_load, upsampler_model_load, base_diffusion, upsampler_diffusion, prompt='a red flower')
save_model2ply(pc, 'D:\\RJdeck\\Metatopia\\meta3d\\flower.ply')