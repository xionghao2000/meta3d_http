from meta3d.services.s3_service import download_file
from meta3d.common import config


download_file('D:\\RJdeck\\Metatopia\\meta3d\\cup_dd.ply', config.BUCKET_NAME,'text_to_3dmodel/cup.ply')
