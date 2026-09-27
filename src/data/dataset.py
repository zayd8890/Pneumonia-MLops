"""
Dataset definitions and data transforms matching the project notebooks.
"""

from typing import Optional, List, Tuple, Union, Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


def get_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    split: str = "train",
) -> transforms.Compose:
    """
    Get data transformations for training, validation, or testing.
    Matches the augmentation and normalization pipelines defined in the notebooks.

    Args:
        image_size: Target image dimension (default: 224).
        mean: Normalization channel means (default: ImageNet mean).
        std: Normalization channel standard deviations (default: ImageNet std).
        split: One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Resize to 256 then center-crop to 224 (matching validation in notebooks)
        resize_dim = int(image_size * 256 / 224) if image_size == 224 else image_size + 32
        return transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class CustomImageDatasetFromDF(Dataset):
    """
    Custom PyTorch Dataset for loading images based on a DataFrame,
    as defined in notebook Copy_of_Untitled0.ipynb.
    Supports single-label or multi-label scenarios.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: Dict[str, int],
        image_path_col: str = "full_image_path",
        label_col: str = "Finding Labels",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            dataframe: DataFrame containing image paths and labels.
            label_map: Dictionary mapping string labels to numeric indices.
            image_path_col: Column name containing the image file path.
            label_col: Column name containing class labels.
            transform: PyTorch transforms to apply to each image.
        """
        self.data = dataframe.reset_index(drop=True)
        self.label_mapping = label_map
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx][self.image_path_col]
        image = Image.open(img_path).convert("RGB")

        label = self.data.iloc[idx][self.label_col]

        # Multi-label case (pipe-separated)
        if isinstance(label, str) and "|" in label:
            labels_list = label.split("|")
            numeric_labels = torch.zeros(len(self.label_mapping), dtype=torch.float32)
            for lbl in labels_list:
                lbl = lbl.strip()
                if lbl in self.label_mapping:
                    numeric_labels[self.label_mapping[lbl]] = 1.0
            target = numeric_labels
        elif isinstance(label, str) and label in self.label_mapping:
            target = self.label_mapping[label]
        elif isinstance(label, (int, float)):
            target = int(label)
        else:
            target = 0

        if self.transform:
            image = self.transform(image)

        return image, target
