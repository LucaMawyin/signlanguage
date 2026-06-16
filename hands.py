import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("runs/pose/train/weights/best.pt")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # No frame
    if not ret:
        print("Error: Can't receive frame.")
        break

    # YOLO hand detection
    results = model(frame, conf=0.01, iou=0.5, verbose=False)
    print(len(results[0].boxes))

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
