import numpy as np
from PIL import Image
import selectivesearch

# Function to extract region proposals using selective search
def extract_region_proposals(image_path, max_proposals=2000):
  img = np.array(Image.open(image_path).convert('RGB'))

  img_lbl, regions = selectivesearch.selective_search(
    img, scale=500, sigma=0.9, min_size=10)

  proposals = []
  for r in regions:
    x, y, w, h = r['rect']
    if w < 20 or h < 20:
      continue
    proposals.append([x, y, x+w, y+h])

  if len(proposals) > max_proposals:
    proposals = proposals[:max_proposals]

  return img, proposals

