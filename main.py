import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms


import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config.config import Config
from utils.extract_regions import extract_region_proposals
from models.RCNN import RCNN

from utils.vs_region_proposls import visualize_region_proposals

transform = transforms.Compose([
  transforms.Resize((227, 227)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Main:
  def __init__(self, model, model_path, device=Config.DEVICE):
    self.device = device
    self.model = model.to(device)
    print(f"Loading model from {model_path}")
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    self.model.eval()
    print("Model loaded successfully")


  def object_detection(self, image_path, save_path=None, show=False):
    print(f"Processing image: {image_path}")
    img, proposals = extract_region_proposals(image_path, max_proposals=150)

    visualize_region_proposals(image_path, proposals)

    print(f"Extracted {len(proposals)} region proposals")

    detected_boxes = []
    detected_scores = []
    detected_classes = []

    for i, proposal in enumerate(proposals):
        x1, y1, x2, y2 = proposal
        cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
        transformed_img = transform(cropped_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_pred, bbox_pred = self.model(transformed_img)
            print(f"Processed proposal {i+1}/{len(proposals)}")

        # Get prediction
        score, class_idx = torch.max(torch.softmax(cls_pred, dim=0), dim=0)
        score = score.item()
        class_idx = class_idx.item()

        # Skip background class
        if class_idx == 0 or score < 0.5:
            continue

        # You could also refine the bounding box here using `bbox_pred` if needed
        detected_boxes.append(proposal)
        detected_scores.append(score)
        detected_classes.append(class_idx)
        print(f"Detected object: {Config.VOC_CLASSES[class_idx]} with score {score:.2f}")

    if not detected_boxes:
        print("No objects detected in the image")
        return

    # Visualize detections
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box, score, class_idx in zip(detected_boxes, detected_scores, detected_classes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        class_name = Config.VOC_CLASSES[class_idx]
        ax.text(x1, y1, f"{class_name}: {score:.2f}", color='white', backgroundcolor='r')

    if save_path:
        plt.savefig(save_path)
        print(f"Output saved to {save_path}")
    if show:
        plt.show()


if __name__ == '__main__':
  model = RCNN(Config.NUM_CLASSES) 
  main = Main(model, 'weights/rcnn_model_with_bbox.pth')
  # main.object_detection('images/plant.jpg', 'outputs/out-plant.png', show=True)
  main.object_detection('data/VOC2007/JPEGImages/000002.jpg', 'outputs/out-002.png', show=True)
