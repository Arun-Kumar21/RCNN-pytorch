import torch
import torch.nn as nn

class RCCNLoss(nn.Module):
  def __init__(self, lambda_reg=1.0) -> None:
    super().__init__()
    self.cls_loss = nn.CrossEntropyLoss()
    self.bbox_loss = nn.SmoothL1Loss(reduction='sum')
    self.lambd_reg = lambda_reg

  def forward(self, cls_pred, bbox_pred, cls_targets, bbox_targets):
    cls_loss = self.cls_loss(cls_pred, cls_targets)

    foreground_idxs = cls_targets > 0

    if foreground_idxs.sum() > 0:
      bbox_pred_fg = bbox_pred[foreground_idxs]
      bbox_targets_fg = bbox_targets[foreground_idxs]

      cls_targets_fg = cls_targets[foreground_idxs]

      bbox_pred_fg_classes = torch.zeros_like(bbox_targets_fg)
      for i in range(bbox_targets_fg.size(0)):
        class_idx = cls_targets_fg[i] - 1
        start_idx = class_idx * 4
        bbox_pred_fg_classes[i] = bbox_pred_fg[i, start_idx: start_idx+4]

      reg_loss = self.bbox_loss(bbox_pred_fg_classes, bbox_targets_fg)
      reg_loss = reg_loss / max(1, foreground_idxs.sum())

    else:
      reg_loss = torch.tensor(0.0).to(cls_loss.device)
    
    total_loss = cls_loss + self.lambd_reg * reg_loss

    return total_loss, cls_loss, reg_loss