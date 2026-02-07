from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision as tv
import pandas as pd

# given (but from where?)
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str = "train"):
        self.data = data # csv containing labels
        self.mode = mode # train or val
        self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize(mean=train_mean, std=train_std)
        ]) #TODO: add more augmentations and differ between train and val trafos
    
    def __len__(self) -> int:
        """Return length of currently loaded data."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.tensor, tuple]: 
        """Returns tuple of image and corresponding label."""
        #we want to use a multilabel setting, so each sample has 2d-Tuple as label
        row = self.data.iloc[index]
        image_path = row.filename
        image_greyscale = imread(Path(image_path))
        image_rgb = gray2rgb(image_greyscale)
        labels = torch.tensor((int(row.crack), int(row.inactive))) #TODO: shall these be ints or floats?
        image_augmented = self._transform(image_rgb)

        return (image_augmented, labels)
