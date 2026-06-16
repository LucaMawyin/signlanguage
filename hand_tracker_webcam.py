import mediapipe as mp
import cv2
from ultralytics import YOLO

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
model = YOLO("runs/pose/train/weights/best.pt")

# Cannot open camera
if not cap.isOpened():
    print("Error: Could not open the video device.")
    exit()

while True:
    

    ret, frame = cap.read()

    # No frame
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Mirror frame and convert from bgr to rgb for mediapipe
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # Draw landmarks if hands found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Final frame
    cv2.imshow('Live Camera Feed', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release windows
cap.release()
cv2.destroyAllWindows()
