import torch
import numpy as np
from utils.iou import calculate_iou

def nms(boxes, scores, threshold=0.7):
    """Apply non-maximum suppression"""
    if len(boxes) == 0:
        return []
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        
        indices = indices[1:][ious <= threshold]
    
    return keep