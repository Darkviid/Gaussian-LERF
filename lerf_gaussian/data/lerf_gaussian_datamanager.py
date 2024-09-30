from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from typing_extensions import TypeVar

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from lerf_gaussian.encoders.image_encoder import BaseImageEncoder
from lerf_gaussian.data.utils.dino_dataloader import DinoDataloader
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from lerf_gaussian.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE


from typing import Type, Literal, Union, Tuple
import torch

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

@dataclass
class SplatfactoDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: SplatfactoDataManager)
    # Include any additional configuration parameters if needed
    # For example, paths to cache directories, etc.
    train_num_rays_per_batch: int = 5000
    """Number of rays per batch to use per training iteration."""
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""

class SplatfactoDataManager(FullImageDatamanager):
    config: SplatfactoDataManagerConfig

    def __init__(
        self,
        config: SplatfactoDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        #self.dino_encoder = kwargs["dino_encoder"]

        # Load and preprocess images
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        # Set up cache directories
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))

        # Initialize DINO and CLIP dataloaders (or encoders), using LERF implementations,
        # it may need changes to work with Gaussian Splatting
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )


    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch with CLIP and DINO embeddings."""

        # Get the next training image and camera
        image_idx = self.train_unseen_cameras.pop(0)
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        data = self.cached_train[image_idx].copy()
        data["image"] = data["image"].to(self.device)

        # Get the corresponding camera
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx

        # Compute CLIP and DINO embeddings
        # For CLIP embeddings
        #image_flat = data["image"].view(-1, data["image"].shape[-1])  # Flatten the image
        #image_batch = next(self.iter_train_image_dataloader)
        #image_batch = example: {'image_idx': tensor([13,  1,  0,  2,  6,  7, 16, 20,  8,  3, 11, 10, 14, 18, 22, 17,  4, 19,
        #15, 21,  5, 12,  9], device='cuda:0'), 'image': tensor([[[[0.6275, 0.4980, 0.4549], shape: N, 
        #  [0.6157, 0.5216, 0.4706],
        #  [0....7],
        #  [0.1412, 0.2000, 0.0863]]]])}

        image_batch = {"image_idx": torch.tensor([image_idx], device=self.device), "image": data["image"].unsqueeze(0)}

        assert self.train_pixel_sampler is not None
        #assert False, "This is a stub, and should be implemented by the user"
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        clip_embeddings, clip_scale = self.clip_interpolator(ray_indices)
        data["clip"] = clip_embeddings
        camera.metadata["clip_scales"] = clip_scale

        # For DINO embeddings
        dino_embeddings = self.dino_dataloader(ray_indices)
        data["dino"] = dino_embeddings

        return camera, data

    def setup_train(self):

        assert self.train_dataset is not None

        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))
        

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )
    

    #Function to calculate CLIP and DINO embeddings from a batch of images
    def get_embeddings(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CLIP and DINO embeddings from a batch of images."""
        embeddings = dict()

        image_batch = {"image_idx": torch.arange(images.shape[0], device=self.device), "image": images}

        assert self.train_pixel_sampler is not None
        #assert False, "This is a stub, and should be implemented by the user"
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        clip_embeddings, clip_scale = self.clip_interpolator(ray_indices)
        dino_embeddings = self.dino_dataloader(ray_indices)

        embeddings["clip"] = clip_embeddings
        embeddings["dino"] = dino_embeddings
        return embeddings