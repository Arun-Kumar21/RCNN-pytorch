# RCNN - Object Detection and Semantic Segmentation

This repository provides a PyTorch implementation of the original [RCNN research paper](https://arxiv.org/pdf/1311.2524), focusing on object detection and semantic segmentation.

## TODO List

- [x] Implement region proposal using selective search.
- [ ] Fine-tune AlexNet on the VOC dataset.
- [ ] Implement a bounding box regressor for precise localization.
- [ ] Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes.
- [ ] Improve accuracy by integrating a VGG-based architecture.

## Region Proposal using Selective Search

We use selective search to extract approximately 2000 region proposals per image for object detection.

The resulting output looks like this:


| Original Image | Region Proposal Image |
|----------------|------------------------|
| ![Original Image](path/to/original_image.jpg) | ![Region Proposal](path/to/region_proposal_image.jpg) |
