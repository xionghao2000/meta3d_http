import flask
import torch
from meta3d.services import Meta3dService
from meta3d.services import PointCloudSampler
from meta3d.services.s3_service import upload_file, get_url
from http import HTTPStatus



# create a flask app
app = flask.Flask(__name__)


@app.route('/test', methods=['POST'])
def test():
    """
    Given the following request body
    {
        "message": "Hello world"
    }
    get the message and return it
    """
    # get the request body
    req = flask.request.get_json()
    # get the message
    message = req['message']

    prompt = message
    model_path = 'D:\\RJdeck\\Metatopia\\meta3d\\meta3d\\models\\'
    ply_path = 'D:\\RJdeck\\Metatopia\\meta3d\\meta3d\\ply\\'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    service = Meta3dService()

    check_model = service.check_model(model_path=model_path)
    base_model, upsampler_model = service.load_model(device=device,model_path=model_path)
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

    file_name = service.save_model2ply(pc, ply_path)

    # upload the file to s3
    object_name = upload_file(file_name)
    # get the url
    url = get_url(object_name)
    
    return flask.jsonify({'url': url}), HTTPStatus.CREATED


# i want run my server on http://localhost:5000
if __name__ == '__main__':
    app.run(
        port=5000,
        debug=True
    )