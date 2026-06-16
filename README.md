Library Depedancy Install Commands:
    pip install ultralytics
    pip install mediapipe==0.10.14
    pip install opencv-python

Order of Operation of File Execution:
    1. create_labels.py : Tracks hand skeletons & creates label txt files
    2. flat_split.py : splits images & labels into train/val data (80/20)
    3. train.py : trains yolo model (30 epochs)