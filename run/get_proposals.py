from utils.extract_regions import extract_region_proposals
from utils.vs_region_proposls import visualize_region_proposals
import os

def get_proposals():
  img_path = 'data/VOC2007/JPEGImages/000004.jpg'

  img, proposals = extract_region_proposals(img_path)
  visualize_region_proposals(img_path, proposals, save_path='outputs/region-proposal/000004.jpg')
