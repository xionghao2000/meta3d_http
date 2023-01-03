from unittest import TestCase

import torch
from point_e.util.point_cloud import PointCloud
from typing import List, Dict, Any, Iterator, Optional

from meta3d.services.meta_3d_service import Meta3dService, PointEService, MachineLearningService


class MockS3Service:
    def __init__(self, should_file_exist) -> None:
        self.counter = 0
        self.should_file_exist = should_file_exist

    def download_file(self, file_path, bucket, object_name):
        self.counter += 1

    def check_exists(self, file_name: str):
        return self.should_file_exist


class MockMachineLearningService(MachineLearningService):
    def load(self, model_path, map_location):
        return None

    def save(self, model, save_path):
        return None


class MockPointEService(PointEService):
    def load_checkpoint(self, base_name, device):
        return None, None

    def model_from_config(self, model_config, device) -> PointEService.Model:
        model = PointEService.Model()
        return model

    def diffusion_from_config(self, diffusion_config):
        return None, None


class MockSampler:
    def sample_batch_progressive(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Iterator[torch.Tensor]:
        for i in range(2):
            yield torch.rand(1, 3)

    def output_to_point_clouds(self, output: torch.Tensor) -> List[PointCloud]:
        return [None]


class MockModel(PointEService.Model):
    def eval(self):
        pass

    def load_state_dict(self, val):
        pass

    def __init__(self, target_output_file_name: str):
        self.target_output_file_name = target_output_file_name

    def write_ply(self, val):
        return self.target_output_file_name


class TestMeta3dService(TestCase):
    def setUp(self):
        self.mock_s3_service_true = MockS3Service(True)
        self.mock_s3_service_false = MockS3Service(False)
        self.mock_pointe_service = MockPointEService()
        self.mock_ml_service = MockMachineLearningService()
        self.mock_sampler = MockSampler()
        self.file_name_to_be_deleted: Optional[str] = None

    def tearDown(self):
        if self.file_name_to_be_deleted is not None:
            import os
            os.remove(self.file_name_to_be_deleted)

    def test_check_model_ture(self):
        service = Meta3dService(s3_service=self.mock_s3_service_true, pointe_service=self.mock_pointe_service)
        service.check_model('D:')
        self.assertEqual(self.mock_s3_service_true.counter, 0)

    def test_check_model_false(self):
        service = Meta3dService(s3_service=self.mock_s3_service_false, pointe_service=self.mock_pointe_service)
        service.check_model('D:')
        self.assertEqual(2, self.mock_s3_service_false.counter)

    def test_create_model(self):
        service = Meta3dService(pointe_service=self.mock_pointe_service)
        service.create_model("")
        self.assertEqual(1, 1)

    def test_load_model(self):
        service = Meta3dService(ml_service=self.mock_ml_service, pointe_service=self.mock_pointe_service)
        service.load_model(" ", " ")
        self.assertEqual(1, 1)

    def test_save_model(self):
        service = Meta3dService(ml_service=self.mock_ml_service, pointe_service=self.mock_pointe_service)
        service.save_model(" ", " ", " ")
        self.assertEqual(1, 1)

    def test_create_diffusion(self):
        service = Meta3dService(pointe_service=self.mock_pointe_service)
        service.create_diffusion("base_300m")
        self.assertEqual(1, 1)

    def test_generate_3d_result(self):
        service = Meta3dService(pointe_service=self.mock_pointe_service)
        service.generate_3d_result(sampler=self.mock_sampler, prompt=" ")
        self.assertEqual(1, 1)

    def test_save_model2ply(self):
        target_model = MockModel("test")
        service = Meta3dService(pointe_service=self.mock_pointe_service)
        self.file_name_to_be_deleted = service.save_model2ply(target_model, "test")
        self.assertIn("test", self.file_name_to_be_deleted)
