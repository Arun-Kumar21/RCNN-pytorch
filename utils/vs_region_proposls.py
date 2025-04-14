import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to visualize extracted proposals from image
def visualize_region_proposals(image_path, proposals, gt_boxes=None, save_path=None):
  img = np.array(Image.open(image_path).convert('RGB'))

  fig, ax = plt.subplots(1)
  ax.imshow(img)

  for i, box in enumerate(proposals[:2000]):  
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  if gt_boxes:
    for gt in gt_boxes:
      x1, y1, x2, y2 = gt['bbox']
      rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
      ax.add_patch(rect)
      ax.text(x1, y1, gt['name'], color='white', backgroundcolor='g')

  if save_path:
    plt.savefig(save_path, bbox_inches='tight')

  plt.show()
