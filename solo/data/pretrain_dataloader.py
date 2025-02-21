# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import pickle
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Type, Union

import torch
import numpy as np
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import functional as F

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class ImageFolderWithSaliency(Dataset):
    def __init__(self, imgs, transform=None):
        super(ImageFolderWithSaliency, self).__init__()
        self.imgs = pickle.load(open(imgs, 'rb'))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.pil_loader(img_path)
        mask = self.npy_loader(mask_path.replace('eyepacs_saliency_map_all', 'eyepacs_with_external_saliency_map'))
        mask = Image.fromarray(np.uint8(mask*255))
        if self.transform is not None:
            img_1, img_2, mask_1, mask_2 = self.transform(img, mask)

        return [img_1, img_2], [mask_1, mask_2], -1

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def npy_loader(self, path):
        with open(path, 'rb') as f:
            img = np.load(f)
            return img


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)


def build_transform_pipeline(dataset, cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.gaussian_blur.prob:
        augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

    if cfg.solarization.prob:
        augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

    if cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

    if cfg.vertical_flip.prob:
        augmentations.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip.prob))

    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.Normalize(mean=mean, std=std))

    augmentations = transforms.Compose(augmentations)
    return augmentations


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        else:
            train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)

    elif dataset == "custom":
        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)
        
    elif dataset == 'eyepacs':
        train_dataset = dataset_with_index(ImageFolderWithSaliency)(train_data_path, transform)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.images
            (
                files,
                _,
            ) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.images = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


class TransformWithSaliency(object):
    def __init__(self, input_size, mean, std, data_aug):
        scale_stu = data_aug['scale_stu']
        scale_tea = data_aug['scale_tea']
        jitter_param = (data_aug['brightness'], data_aug['contrast'], data_aug['saturation'], data_aug['hue'])
        degree = data_aug['degrees']

        self.resized_crop_stu = transforms.RandomResizedCrop(input_size, scale=scale_stu)
        self.color_jitter_stu = transforms.RandomApply([transforms.ColorJitter(*jitter_param)], p=0.8)
        self.grayscale_stu = transforms.RandomGrayscale(p=0.2)
        self.gaussian_blur_stu = transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0)
        self.rotation_stu = transforms.RandomRotation(degree)
        self.p_rotation_stu = 0.8
        self.p_hflip_stu = 0.5
        self.p_vflip_stu = 0.5

        self.resized_crop_tea = transforms.RandomResizedCrop(input_size, scale=scale_tea)
        self.color_jitter_tea = transforms.RandomApply([transforms.ColorJitter(*jitter_param)], p=0.8)
        self.grayscale_tea = transforms.RandomGrayscale(p=0.2)
        self.gaussian_blur_tea = transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1)
        self.rotation_tea = transforms.RandomRotation(degree)
        self.solarize_tea = transforms.RandomApply([Solarization()], p=0.2)
        self.p_rotation_tea = 0.8
        self.p_hflip_tea = 0.5
        self.p_vflip_tea = 0.5

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img, mask):
        img_stu, mask_stu = self.resized_crop_with_mask(self.resized_crop_stu, img, mask)
        img_stu = self.color_jitter_stu(img_stu)
        img_stu = self.grayscale_stu(img_stu)
        img_stu = self.gaussian_blur_stu(img_stu)
        img_stu, mask_stu = self.rotation_with_mask(self.rotation_stu, img_stu, mask_stu, self.p_rotation_stu)
        img_stu, mask_stu = self.horizontal_flip_with_mask(img_stu, mask_stu, self.p_hflip_stu)
        img_stu, mask_stu = self.vertical_flip_with_mask(img_stu, mask_stu, self.p_vflip_stu)
        img_stu, mask_stu = self.to_tensor(img_stu), self.to_tensor(mask_stu)
        img_stu = self.normalize(img_stu)

        img_tea, mask_tea = self.resized_crop_with_mask(self.resized_crop_tea, img, mask)
        img_tea = self.color_jitter_tea(img_tea)
        img_tea = self.grayscale_tea(img_tea)
        img_tea = self.gaussian_blur_tea(img_tea)
        img_tea = self.solarize_tea(img_tea)
        img_tea, mask_tea = self.rotation_with_mask(self.rotation_tea, img_tea, mask_tea, self.p_rotation_tea)
        img_tea, mask_tea = self.horizontal_flip_with_mask(img_tea, mask_tea, self.p_hflip_tea)
        img_tea, mask_tea = self.vertical_flip_with_mask(img_tea, mask_tea, self.p_vflip_tea)
        img_tea, mask_tea = self.to_tensor(img_tea), self.to_tensor(mask_tea)
        img_tea = self.normalize(img_tea)

        return img_stu, img_tea, mask_stu, mask_tea

    def resized_crop_with_mask(self, tf, img, mask):
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(img, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        mask = F.resized_crop(mask, i, j, h, w, tf.size, tf.interpolation)
        return img, mask

    def rotation_with_mask(self, tf, img, mask, p):
        assert isinstance(tf, transforms.RandomRotation)
        if random.random() < p:
            angle = tf.get_params(tf.degrees)
            img = F.rotate(img, angle, tf.interpolation, tf.expand, tf.center, tf.fill)
            mask = F.rotate(mask, angle, tf.interpolation, tf.expand, tf.center, tf.fill)
        return img, mask

    def horizontal_flip_with_mask(self, img, mask, p):
        if random.random() < p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

    def vertical_flip_with_mask(self, img, mask, p):
        if random.random() < p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask
