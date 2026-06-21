# YOLO Hand Tracking Project

## Description

This project implements a YOLO model to detect user hand gestures of the ASL alphabet and translates it to speech.

### Logistical Specs

The model was trained on 5600 images, with each gesture composing of 200 images, using a YOLOv26m model for 200 epochs with 12 workers, batch size 16, and a patience of 40. The specs of the PC that ran the initial model are as follows:

- CPU: Intel Core i5 13600kf
- GPU: Nvidia RTX 3080 Ti 15G VRAM
- RAM: 48G 3200MHz

## Limitations

The MediaPipe, and subsequently YOLO model was unable to successfully register the hand gestures for both space and backspace. As a solution the program auto segments words instead of relying on manual space signaling, and deleting is performed via the backspace key.

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
pip install wordsegment
pip install pyttsx3
```

## File Execution Order (If Training New Model)

### 1. `create_labels.py`

- Tracks hand skeleton landmarks
- Generates YOLO `.txt` label files

### 2. `yolo_model.py`

- Trains the YOLOv26m model for **200 epochs**

**Note:**
Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
