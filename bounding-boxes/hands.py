import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("runs/hand_pose/weights/best.pt").to("cuda")

text_str = ""
last_class = None

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # No frame
    if not ret:
        break

    # YOLO hand detection
    results = model(frame, conf=0.05, iou=0.5, verbose=False)

    r = results[0]

    # Get best detection (if any exist)
    if r.boxes is not None and len(r.boxes) > 0:
        cls_id = int(r.boxes.cls[0])
        class_name = model.names[cls_id]

        # only print if changed
        if class_name != last_class:
            if class_name == "del":
                text_str = text_str[:-1]
            elif class_name == "space":
                text_str = " "
            else:
                text_str += class_name
                
            print("Current string:", text_str)
            last_class = class_name

    # force higher confidence visibility
    annotated_frame = results[0].plot(
        conf=True,
        line_width=2,
        labels=True
    )
    
    # Final frame
    cv2.imshow('Hand Tracking', annotated_frame)
    
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
