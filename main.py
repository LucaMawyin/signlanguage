import cv2
import wordsegment
from wordsegment import segment
from ultralytics import YOLO

wordsegment.load()
cap = cv2.VideoCapture(0)
model = YOLO("runs/hand_pose/weights/best.pt").to("cuda")

raw_text_str = ""
segmented_text = ""
last_class = None
stable_count = 0
required_frames = 10

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
    results = model(frame, conf=0.5, verbose=False)

    r = results[0]

    # Get best detection (if any exist)
    if r.boxes is not None and len(r.boxes) > 0:
        best_idx = r.boxes.conf.argmax()

        cls_id = int(r.boxes.cls[best_idx])
        conf = float(r.boxes.conf[best_idx])
        class_name = model.names[cls_id]


        if class_name == last_class:
            stable_count += 1
        else:
            stable_count = 0
            last_class = class_name

        # Only print on stabilized frames
        if stable_count == required_frames:

            if class_name == "del" or class_name == "space":
                continue
            else:
                raw_text_str += class_name

                words = segment(raw_text_str) # Infer where spaces go
                segmented_text = " ".join(words)

            last_class = class_name

    # Force higher confidence visibility
    annotated_frame = results[0].plot(
        conf=True,
        line_width=2,
        labels=True
    )

    key = cv2.waitKey(1)
    
    # Backspace key
    if key == 8:  # Windows Backspace
        raw_text_str = raw_text_str[:-1]

        # Segment
        words = segment(raw_text_str)
        segmented_text = " ".join(words)
    
    # Quit
    if key & 0xFF == ord('q'):
        break

    cv2.putText(
        annotated_frame,
        segmented_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Final frame
    cv2.imshow('Hand Tracking', annotated_frame)

# Release capture
cap.release()
cv2.destroyAllWindows()
