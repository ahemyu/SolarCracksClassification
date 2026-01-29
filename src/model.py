import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module): 
    """nn Module representing a Residual Block.""" 
    def __init__(self, in_channels: int, out_channels: int, stride: int, filter_size: int = 3):
        super().__init__()
        #define main path layers
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter_size, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, filter_size, 1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        #define skip paths
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, input_tensor: torch.tensor): 
        #send input through main path
        main_path = self.main(input_tensor)
        #send input through skip path
        skip_path = self.skip(input_tensor)
        #add them up and apply RELU on the result
        return F.relu(main_path + skip_path)


class Model(nn.Module):
    """Variant of Resnet Architecture for multilabel classification."""
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, padding=3), #TODO: not sure if padding here is needed
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
    def forward(self, input_tensor: torch.tensor):
        #send the input through all of the layers
        return self.layers(input_tensor)


# # test code
# from data import ChallengeDataset
# import pandas as pd
# data = pd.read_csv("labels.csv", sep=";")
# dataset = ChallengeDataset(data)
# sample = dataset[0][0].unsqueeze(0)
# model = Model()
# res = model.forward(sample)
# print(res)