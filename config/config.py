import torch

class Config:
  BATCH_SIZE = 32
  EPOCHS = 100
  NUM_CLASSES = 21
  LEARNING_RATE = 0.001
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  IOU_THRESHOLD = 0.5
  VOC_CLASSES =  [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
  ]