from dataclasses import dataclass, field
from typing import Type
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
#from nerfstudio.model_components.base_model import ModelConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from lerf_gaussian.data.lerf_gaussian_datamanager import SplatfactoDataManagerConfig, SplatfactoDataManager
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig
from lerf_gaussian.lerf_gaussian import SplatfactoModelConfig
from lerf_gaussian.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder


@dataclass
class GaussianLERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for Gaussian LERF pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussianLERFPipeline)
    """Target class to instantiate"""
    datamanager: SplatfactoDataManagerConfig = SplatfactoDataManagerConfig()
    """Specifies the data manager config"""
    model: ModelConfig =  SplatfactoModelConfig()
    """Specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """Specifies the vision-language network config"""

class GaussianLERFPipeline(VanillaPipeline):
    def __init__(
        self,
        config: GaussianLERFPipelineConfig,
        device: str,
        test_mode: str = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        print(f"GaussianLERFPipeline: test_mode={test_mode}")
        self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: SplatfactoDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
        )

        self.datamanager.to(device)

        self._model = config.model.setup(
            device=device,
            datamanager=self.datamanager,
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

        self.model.to(device)

        #Set a reference of the data manager to the model
        self.model.datamanager = self.datamanager