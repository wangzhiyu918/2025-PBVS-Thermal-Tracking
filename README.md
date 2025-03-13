# 2025-CVPR-PBVS-Thermal-Pedestrian-Tracking

## Team Member

Zhiyu Wang, Weiqing Lu, Puhong Duan, Bin Sun, Xudong Kang, Shutao Li

## Method

We use YOLOv8 as the detector and ByteTrack as the tracker for thermal pedestrian tracking.

## Usage

Please follow [mmtrack](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md) and [ultralytics](https://github.com/ultralytics/ultralytics) to install the requirements.

```
# 1. We first train a YOLOv8s model to detect person in the thermal image
# The pretrained weights are located in the ./ultralytics/runs/detect/train/weights
cd ultralytics
python scripts/train.py

# 2. We use the trained YOLOv8s model to generate detection results (i.e., ./predictions.json)
cd ultralytics
python scripts/infer.py

# 3. We use mmtracking to track the pedestrian
# modify the image dir in utils/infer_results.py
# the results are located in the ./preds
python utils/infer_results.py

```

