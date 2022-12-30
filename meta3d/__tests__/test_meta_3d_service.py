from meta3d.services.meta_3d_service import Meta3dService, PointEService
from unittest import TestCase

class MockS3Service:
    def __init__(self, should_file_exist) -> None:
        self.counter = 0
        self.should_file_exist = should_file_exist

    def download_file(self, file_path, bucket, object_name):
        self.counter += 1
    
    def check_exists(self, file_name: str):
        return self.should_file_exist


class MockPointEService(PointEService):
    def load_checkpoint(self, base_name, device):
        return None, None

    def model_from_config(self, model_config, device) -> PointEService.Model:
        model = PointEService.Model()
        return model
        

class TestMeta3dService(TestCase):
    def setUp(self):
        self.mock_s3_service_true = MockS3Service(True)
        self.mock_s3_service_false = MockS3Service(False)
        self.mock_pointe_service = MockPointEService()

    def test_check_model_ture(self):
        service = Meta3dService(s3_service=self.mock_s3_service_true)
        service.check_model('D:')
        self.assertEqual(self.mock_s3_service_true.counter, 0)

    def test_check_model_false(self):
        service = Meta3dService(s3_service=self.mock_s3_service_false)
        service.check_model('D:')
        self.assertEqual(2, self.mock_s3_service_false.counter)

    def test_create_model(self):
        service = Meta3dService(pointe_service=self.mock_pointe_service)
        service.create_model("")
        self.assertEqual(1, 1)