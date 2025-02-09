from os import path, walk

import torch

from torch.utils.data import Dataset

from torchvision.io import decode_image

from torchvision.transforms.v2 import (
    Transform,
    Compose,
    Resize,
    CenterCrop,
    ToDtype,
)


class ImageFolder(Dataset):
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    IMAGE_MODE = "RGB"

    def __init__(
        self,
        root_path: str,
        upscale_ratio: int,
        target_resolution: int,
        pre_transformer: Transform | None = None,
    ):
        if upscale_ratio not in (2, 4, 8):
            raise ValueError(
                f"Upscale ratio must be either 2, 4, or 8, {upscale_ratio} given."
            )

        if target_resolution % upscale_ratio != 0:
            raise ValueError(
                f"Target resolution must divide evenly into upscale_ratio."
            )

        image_paths = [
            path.join(folder_path, filename)
            for folder_path, _, filenames in walk(root_path)
            for filename in filenames
            if self.has_image_extension(filename)
        ]

        input_transformer = Compose(
            [
                Resize(target_resolution // upscale_ratio),
                CenterCrop(target_resolution // upscale_ratio),
                ToDtype(torch.float32, scale=True),
            ]
        )

        target_transformer = Compose(
            [
                Resize(target_resolution),
                CenterCrop(target_resolution),
                ToDtype(torch.float32, scale=True),
            ]
        )

        self.image_paths = image_paths
        self.pre_transformer = pre_transformer
        self.input_transformer = input_transformer
        self.target_transformer = target_transformer

    @classmethod
    def has_image_extension(cls, filename: str) -> bool:
        _, extension = path.splitext(filename)

        return extension in cls.ALLOWED_EXTENSIONS

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = decode_image(image_path, mode=self.IMAGE_MODE)

        if self.pre_transformer:
            image = self.pre_transformer(image)

        x = self.input_transformer(image)
        y = self.target_transformer(image)

        return x, y

    def __len__(self):
        return len(self.image_paths)
