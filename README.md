# YOLO Hand Tracking Project

## CAUTION

The YOLO folder contains:

- 5600 training images
- 5600 label files
- 5600 parsed images

User discretion is advised when opening.

## Dependencies

Install required libraries:

```
pip install ultralytics
pip install mediapipe==0.10.14
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## File Execution Order (If Training New Model)

### 1. `create_labels.py`

- Tracks hand skeleton landmarks
- Generates YOLO `.txt` label files

### 2. `yolo_model.py`

- Trains the YOLOv26s model for **300 epochs**

**Note:**
Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
