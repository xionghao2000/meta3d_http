import torch
from meta3d.services.meta_3d_service import \
    create_model, save_model, load_model, create_diffusion, generate_3dmodel, save_model2ply
from meta3d.services.s3_service import upload_file, get_url
from http import HTTPStatus

# message = 'a apple'

# # initialize the model
# base_name = 'base40M-textvec'
# # select the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # select the file path
# base_model_path = 'D:\\RJdeck\\Metatopia\\meta3d\\base_model.pt'
# unsample_model_path = 'D:\\RJdeck\\Metatopia\\meta3d\\upsample_model.pt'
ply_filename = 'D:\\RJdeck\\Metatopia\\meta3d\\apple.ply'
# # load the model
# base_model_load, upsampler_model_load = load_model(device,base_model_path ,unsample_model_path )
# # create the diffusion
# base_diffusion, upsampler_diffusion = create_diffusion(base_name)
# # generate the 3d model
# pc = generate_3dmodel(device, base_model_load, upsampler_model_load, base_diffusion, upsampler_diffusion, message)
# # save the model
# save_model2ply(pc, ply_filename)

# upload the file to s3
object_name = upload_file(ply_filename)
# get the url
url = get_url(object_name)
print(url)