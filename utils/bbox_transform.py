import torch

def bbox_transform(proposal_roi, gt_roi):
  # roi -> region of interest
  proposal_width = proposal_roi[:, 2] - proposal_roi[:, 0] + 1.0
  proposal_height = proposal_roi[:, 3] - proposal_roi[:, 1] + 1.0
  proposals_ctr_x = proposal_roi[:, 0] + 0.5 * proposal_width
  proposals_ctr_y = proposal_roi[:, 1] + 0.5 * proposal_height

  gt_width = gt_roi[:, 2] - gt_roi[:, 0] + 1.0
  gt_height = gt_roi[:, 3] - gt_roi[:, 1] + 1.0
  gt_ctr_x = gt_roi[:, 0] + 0.5 * gt_width
  gt_ctr_y = gt_roi[:, 1] + 0.5 * gt_height

  targets_dx = (gt_ctr_x - proposals_ctr_x) / proposal_width
  targets_dy = (gt_ctr_y - proposals_ctr_y) / proposal_height
  targets_dw = torch.log(gt_width / proposal_width)
  targets_dh = torch.log(gt_height / proposal_height)

  targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
  return targets 


def bbox_transform_inv(boxes, deltas):
  if boxes.size == 0:
    return torch.zeros((0, 4), device=deltas.device)
  
  boxes = boxes.to(deltas.device)
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]
  
  pred_ctr_x = dx * widths + ctr_x
  pred_ctr_y = dy * heights + ctr_y
  pred_w = torch.exp(dw) * widths
  pred_h = torch.exp(dh) * heights

  pred_boxes = torch.zeros_like(deltas)
  pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
  pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
  pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
  pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes
