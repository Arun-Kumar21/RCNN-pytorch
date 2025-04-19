import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms


import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config.config import Config

from utils.extract_regions import extract_region_proposals
from utils.bbox_transform import bbox_transform_inv
from utils.nms import nms

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
    img, proposals = extract_region_proposals(image_path, max_proposals=2000)

    visualize_region_proposals(image_path, proposals)

    print(f"Extracted {len(proposals)} region proposals")

    all_boxes = []
    all_scores = []
    all_classes = []
    all_refined_boxes = []

    for i, proposal in enumerate(proposals):
        x1, y1, x2, y2 = proposal
        cropped_img = Image.fromarray(img).crop((x1, y1, x2, y2))
        transformed_img = transform(cropped_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_pred, bbox_pred = self.model(transformed_img)
            print(f"Processed proposal {i + 1}/{len(proposals)}")

        probs = torch.softmax(cls_pred, dim=1)

        for class_idx in range(1, Config.NUM_CLASSES):  # Skip background class
            score = probs[0, class_idx].item()
            if score < 0.5:
                continue

            bbox_offset = bbox_pred[0, (class_idx - 1) * 4: class_idx * 4]
            proposal_tensor = torch.tensor([proposal], dtype=torch.float32).to(self.device)
            refined_box = bbox_transform_inv(proposal_tensor, bbox_offset.unsqueeze(0))[0]
            refined_box = refined_box.cpu().numpy().tolist()

            img_height, img_width = img.shape[:2]
            refined_box[0] = max(0, min(refined_box[0], img_width - 1))
            refined_box[1] = max(0, min(refined_box[1], img_height - 1))
            refined_box[2] = max(0, min(refined_box[2], img_width - 1))
            refined_box[3] = max(0, min(refined_box[3], img_height - 1))

            all_boxes.append(proposal)
            all_scores.append(score)
            all_classes.append(class_idx)
            all_refined_boxes.append(refined_box)

    final_boxes = []
    final_scores = []
    final_classes = []
    final_refined_boxes = []

    for class_id in range(1, Config.NUM_CLASSES):
        indices = [i for i, cls in enumerate(all_classes) if cls == class_id]
        if not indices:
            continue

        class_boxes = [all_refined_boxes[i] for i in indices]
        class_scores = [all_scores[i] for i in indices]

        keep_indices = nms(class_boxes, class_scores, threshold=0.3)

        for idx in keep_indices:
            orig_idx = indices[idx]
            final_boxes.append(all_boxes[orig_idx])
            final_scores.append(all_scores[orig_idx])
            final_classes.append(all_classes[orig_idx])
            final_refined_boxes.append(all_refined_boxes[orig_idx])

    if not final_boxes:
        print("No objects detected in the image")
        return

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box, refined_box, score, class_idx in zip(final_boxes, final_refined_boxes, final_scores, final_classes):
        # Original proposal box (dashed red)
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        # Refined box (green)
        rx1, ry1, rx2, ry2 = refined_box
        refined_rect = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(refined_rect)

        class_name = Config.VOC_CLASSES[class_idx]
        ax.text(rx1, ry1, f"{class_name}: {score:.2f}", color='white', backgroundcolor='g')

    if save_path:
        plt.savefig(save_path)
        print(f"Output saved to {save_path}")
    if show:
        plt.show()


if __name__ == '__main__':
  model = RCNN(Config.NUM_CLASSES) 
  main = Main(model, 'weights/rcnn_model_with_bbox.pth')
  main.object_detection('images/plant.jpg', 'outputs/out-plant.png', show=True)
  # main.object_detection('data/VOC2007/JPEGImages/000045.jpg', 'outputs/out-045.png', show=True)
