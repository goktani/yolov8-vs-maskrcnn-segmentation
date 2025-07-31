# YOLOv8-Seg vs Mask R-CNN: Instance & Semantic Segmentation Comparison

A hands-on comparison between YOLOv8-Seg and Mask R-CNN for object segmentation tasks. This repository explores instance and (pseudo) semantic segmentation using the two state-of-the-art models, visualizing and comparing their prediction results on the same image.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Notes](#notes)
- [References](#references)
- [License](#license)

---

## Overview

This project demonstrates and visualizes the performance differences between [YOLOv8-Seg](https://docs.ultralytics.com/tasks/segment/) (from Ultralytics) and [Mask R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html) (from torchvision) on both instance and semantic segmentation tasks.  
You can run both models on the same image and compare:

- **Instance Segmentation:** Each object instance is segmented separately.
- **Semantic Segmentation (pseudo):** Aggregated instance masks generate class-level segmentation.

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/yolov8-vs-maskrcnn-segmentation.git
    cd yolov8-vs-maskrcnn-segmentation
    ```

2. **Set up your environment and install dependencies**
    ```bash
    pip install torch torchvision ultralytics matplotlib opencv-python
    ```

3. **Download YOLOv8n-seg model weights**
    - Download `yolov8n-seg.pt` from [here](https://github.com/ultralytics/ultralytics).
    - Place it in the project directory.

4. **Prepare a sample image**
    - Add your test image and update the `image_path` variable in the script with its path.

---

## Usage

1. **Edit the script**
    - Change the `image_path` in the script to your test image.

2. **Run the script**
    ```bash
    python segmentation_comparison.py
    ```

    This will generate and display a figure showing both instance and (pseudo) semantic segmentation results side-by-side for YOLOv8-Seg and Mask R-CNN.

---

## Sample Output

Below is an example output comparing YOLOv8-Seg and Mask R-CNN, both on instance and semantic segmentation tasks:

![Sample Output](output.png)

---

## Notes

- **Semantic Segmentation (pseudo):** Since both models are designed for instance segmentation, "semantic" segmentation is produced by merging all instance masks for each model.
- **Performance:** For large images or many detected instances, processing might take longer.
- **Device:** The provided script runs on CPU by default. For faster processing, consider moving models and tensors to GPU.

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Mask R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)
- [Image Segmentation at Wikipedia](https://en.wikipedia.org/wiki/Image_segmentation)

---

## License

This project is intended for educational and research purposes.

---

**Enjoy experimenting with segmentation models!**
