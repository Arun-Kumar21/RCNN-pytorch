import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class RCNN(nn.Module):
  def __init__(self, num_classes) -> None:
    super(RCNN, self).__init__()

    self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)

    self.alexnet.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6 , 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes)
    )

  def forward(self, x):
    return self.alexnet(x)

