import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

class RCNN(nn.Module):
  def __init__(self, num_classes) -> None:
    super(RCNN, self).__init__()

    alexnet_model = alexnet(pretrained=True)

    self.features = alexnet_model.features
    self.avgpool = alexnet_model.avgpool

    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6 , 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
    )

    self.cls_score = nn.Linear(4096, num_classes)

    self.bbox_pred = nn.Linear(4096, 4 * (num_classes -1))

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x)
    x = self.classifier(x)

    cls_score = self.cls_score(x)
    bbox_pred = self.bbox_pred(x)

    return cls_score, bbox_pred
