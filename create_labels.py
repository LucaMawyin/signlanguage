import mediapipe as mp
import cv2
import os
import time
import string

# Dictionary with our keys
chars = list(string.ascii_lowercase) + ["space", "del"]
key_classes = {char: i for i, char in enumerate(chars)}

# Path setup
root_dir = "yolo-data-temp/code/data/"
image_dir = root_dir + "images/train"
label_dir = root_dir + "labels/train"
temp_dir = root_dir + "mapped_images_output/"
os.makedirs(label_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.2,
)

# Label each image in image dir
for img_name in os.listdir(image_dir):
    if not img_name.endswith(".jpg"):
        continue

    # Load image 
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w, _ = image.shape

    # Process hand using mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Image copy for visualization
    debug_image = image.copy()

    label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:

        # Getting class for hand gesture
        key = img_name.split("_")[0].lower()
        class_id = key_classes.get(key)

        if class_id is None:
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # Bounding box for hand
                cv2.rectangle(
                    debug_image,
                    (int(x_min * image.shape[1]), int(y_min * image.shape[0])),
                    (int(x_max * image.shape[1]), int(y_max * image.shape[0])),
                    (0, 255, 0),
                    2
                )

                class_name = chars[class_id]

                cv2.putText(
                    debug_image,
                    class_name,
                    (int(x_min * image.shape[1]), int(y_min * image.shape[0]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                kpts = []
                for lm in hand_landmarks.landmark:
                    kpts.extend([lm.x, lm.y, 1.0])


                label = [class_id, x_center, y_center, width, height] + kpts

                f.write(" ".join(map(str, label)) + "\n")

    cv2.imwrite(os.path.join(temp_dir, img_name), debug_image)
