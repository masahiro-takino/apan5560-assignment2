import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN architecture required for Assignment 2.

    Input: RGB image of shape 3 x 64 x 64
    Conv1: 16 filters, 3x3 kernel, stride 1, padding 1
    MaxPool: 2x2 kernel, stride 2
    Conv2: 32 filters, 3x3 kernel, stride 1, padding 1
    MaxPool: 2x2 kernel, stride 2
    FC1: 100 units
    FC2: 10 output classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name: str):
    model_name = model_name.upper()

    if model_name == "CNN":
        return CNN()

    raise ValueError("Unsupported model_name. Available option: CNN")
