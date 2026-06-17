import mediapipe as mp
import cv2
import os
import string

# Dictionary with our keys
chars = list(string.ascii_lowercase) + ["space", "del"]
key_classes = {char: i for i, char in enumerate(chars)}

# Path setup
root_dir = "yolo-data/code/data/"
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
    min_detection_confidence=0.15,
)

count = 0

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

        key = img_name.split("_")[0].lower()
        class_id = key_classes.get(key)

        if class_id is None:
            continue

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                    
                # Draw hand landmarks to output
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # x and y values for hand landmarks
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # Pad hand box
                pad = 0.15

                box_w = x_max - x_min
                box_h = y_max - y_min

                x_min = max(0, x_min - pad * box_w)
                x_max = min(1, x_max + pad * box_w)

                y_min = max(0, y_min - pad * box_h)
                y_max = min(1, y_max + pad * box_h)

                # YOLO format (normalized)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # Write YOLO label
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                # Draw bbox for visualization
                h, w, _ = image.shape

                cv2.rectangle(
                    debug_image,
                    (int(x_min * w), int(y_min * h)),
                    (int(x_max * w), int(y_max * h)),
                    (0, 255, 0),
                    2
                )

                # Label text
                class_name = chars[class_id]

                cv2.putText(
                    debug_image,
                    class_name,
                    (int(x_min * w), int(y_min * h) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


    cv2.imwrite(os.path.join(temp_dir, img_name), debug_image)

    # Print a message every 100 images
    count += 1
    if count % 100 == 0:
        print(f"[INFO] Processed {count}/{len(os.listdir(image_dir))} images")
