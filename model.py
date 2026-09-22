import torch.nn as nn

class MNIST_Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d( # 28x28
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(2), # 14x14

        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),
        nn.MaxPool2d(2),
    )

    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x