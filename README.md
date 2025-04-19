# RCNN - Object Detection and Semantic Segmentation

This repository provides a PyTorch implementation of the original [RCNN research paper](https://arxiv.org/pdf/1311.2524), focusing on object detection and semantic segmentation.

## TODO List

- [x] Implement region proposal using selective search.
- [x] Fine-tune AlexNet on the VOC dataset.
- [x] Implement a bounding box regressor for precise localization.
- [x] Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes.
- [ ] Improve accuracy by integrating a VGG-based architecture.

## Region Proposal using Selective Search

We use selective search to extract approximately 2000 region proposals per image for object detection.

The resulting output looks like this:


| Original Image | Region Proposal Image |
|----------------|------------------------|
| ![Original Image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/000002.jpg) | ![Region Proposal](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/out-0002.jpg) |
| ![Original Image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/000004.jpg) | ![Region Proposal](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/out-0004.jpg) |


## Object Detection using AlexNet with Bounding Box Regressor

We fine-tune AlexNet on the VOC dataset to classify objects and employ a bounding box regressor for enhanced localization accuracy.

The resulting output looks like this:

| Original Image | Detected Objects |
|----------------|------------------|
| ![Original Image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/000002.jpg) | ![Detected Objects](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/out-002.png) |
| ![Original Image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/plant.jpg) | ![Detected Objects](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/out-plant.png) |

## Non-Maximum Suppression (NMS) for Reducing Overlapping Boxes

Non-Maximum Suppression (NMS) is applied to eliminate overlapping bounding boxes, ensuring that only the most confident prediction is retained for each detected object.

The resulting output looks like this:

| Original Image | After NMS |
|----------------|-----------|
| ![Original Image](https://myjournalbucket-arun.s3.eu-north-1.amazonaws.com/plant.jpg) | ![After NMS](https://raw.githubusercontent.com/Arun-Kumar21/RCNN-pytorch/refs/heads/master/outputs/out-plant.png) |


## Contributors

This project is maintained and developed by:
- **Arun Kumar**
- **Deepak Diwan**
- **Satyam Kumar**
- **Mansi Aggarwal**