import cv2
import wordsegment
from wordsegment import segment
from ultralytics import YOLO
import pyttsx3
import threading

wordsegment.load()
cap = cv2.VideoCapture(0)
model = YOLO("runs/hand_pose/weights/best.pt").to("cuda")

# TTS engine
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) 
    engine.say(text)
    engine.runAndWait()
    del engine

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

    # Get best detection
    if r.boxes is not None and len(r.boxes) > 0:
        best_idx = r.boxes.conf.argmax()

        cls_id = int(r.boxes.cls[best_idx])
        conf = float(r.boxes.conf[best_idx])
        class_name = model.names[cls_id]

        # Stabilize on 10 frames
        if class_name == last_class:
            stable_count += 1
        else:
            stable_count = 0
            last_class = class_name

        # Only process on stabilized frame
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
    if key == 8: 
        raw_text_str = raw_text_str[:-1]

        # Segment
        words = segment(raw_text_str)
        segmented_text = " ".join(words)

    # Enter key
    if key == 13: 
        threading.Thread(target=speak, args=(segmented_text,), daemon=True).start() # New thread so no freeze
        segmented_text = ""
        raw_text_str = ""
    
    # Quit
    if key & 0xFF == ord('q'):
        break


    # Frame text
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
