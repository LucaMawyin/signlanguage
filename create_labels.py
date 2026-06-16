import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

input_folder = "yolo-data-temp/code/data/images/train"
output_label_folder = "yolo-data-temp/code/data/labels/train"

os.makedirs(output_label_folder, exist_ok=True)

class_names = sorted([
    d for d in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, d))
])

class_map = {name: i for i, name in enumerate(class_names)}

for class_name in class_names:
    class_path = os.path.join(input_folder, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        for hand_landmarks in result.multi_hand_landmarks:

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            # bounding box from keypoints
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = (x_max - x_min)
            height = (y_max - y_min)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
                keypoints.append(2)  # visibility

            class_id = class_map[class_name]

            label_line = [class_id, x_center, y_center, width, height] + keypoints

            label_path = os.path.join(
                output_label_folder,
                img_name.replace(".jpg", ".txt")
            )

            with open(label_path, "w") as f:
                f.write(" ".join(map(str, label_line)))

print("Done converting to YOLO Pose format")