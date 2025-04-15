import os
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
from torchvision.transforms import transforms

from config.config import Config

from utils.VOC_annotation_parser import parse_voc_annotation
from utils.extract_regions import extract_region_proposals
from utils.iou import calculate_iou


class RCNNDataset(Dataset):
  def __init__(self, voc_root, transform=None, max_proposals=128, images=500) -> None:
    super().__init__()
    self.voc_root = voc_root
    self.transform = transform
    self.max_proposals = max_proposals
    
    self.img_dir = os.path.join(voc_root, 'JPEGImages')
    self.annotation_dir = os.path.join(voc_root, 'Annotations')

    self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
    self.image_files = self.image_files[:images]

    self.samples = []
    self.prepare_dataset()
    
  def prepare_dataset(self):
    for image in self.image_files:
      image_id = os.path.splitext(image)[0]
      image_path = os.path.join(self.img_dir, image)
      xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

      gt_objects, (width, height) = parse_voc_annotation(xml_path)

      img, proposals = extract_region_proposals(image_path) 
    
      positive_samples = []
      negative_samples = []

      for proposal in proposals:
        max_iou = 0
        best_gt = None
        
        for gt_object in gt_objects:
          iou = calculate_iou(proposal, gt_object['bbox'])
          if iou > max_iou:
            max_iou = iou
            best_gt = gt_object

        # If max_iou greater than threshold it is positive sample
        if max_iou > Config.IOU_THRESHOLD:
          class_idx = Config.VOC_CLASSES.index[best_gt['name']]
          positive_samples.append((proposal, class_idx))

        elif max_iou < 0.3:
          negative_samples.append((proposal, 0)) # Background class

      random.shuffle(negative_samples)
      negative_samples = negative_samples[:len(positive_samples) * 3]  # 3:1 ratio

      samples = positive_samples + negative_samples
      random.shuffle(samples)
      
      if len(samples) > self.max_proposals:
        samples = samples[:self.max_proposals]
      
      for proposal, label in samples:
        self.samples.append((image_path, proposal, label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    img_path, proposal, label = self.samples[index]
    
    img = Image.open(img_path)
    x1, y1, x2, y2 = proposal
    cropped_img = img.crop((x1, y1, x2, y2))
    
    if self.transform:
      cropped_img = self.transform(cropped_img)
    
    return cropped_img, label


transform = transforms.Compose({
  transforms.Resize((227, 227)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
})
  
dataset = RCNNDataset(voc_root='data/VOC2007', transform=transforms, max_proposals=150)
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)

