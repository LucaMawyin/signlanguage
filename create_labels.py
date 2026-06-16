import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

root_dir = "yolo-data-temp"

input_folder = root_dir + "/code/data/images/train"
output_label_folder = root_dir + "/code/data/labels/train"

os.makedirs(output_label_folder, exist_ok=True)

# -----------------------
# Build class map
# -----------------------
class_names = set()

for img_name in os.listdir(input_folder):
    if "_" not in img_name:
        continue
    class_name = img_name.split("_", 1)[0]
    class_names.add(class_name)

class_names = sorted(list(class_names))
class_map = {name: i for i, name in enumerate(class_names)}

print("Class map:", class_map)

# -----------------------
# Process images
# -----------------------
for img_name in os.listdir(input_folder):

    if "_" not in img_name:
        continue

    class_name = img_name.split("_", 1)[0]

    if class_name not in class_map:
        continue

    img_path = os.path.join(input_folder, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    class_id = class_map[class_name]

    label_lines = []

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            h, w, _ = img.shape

            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append(float(lm.x))
                keypoints.append(float(lm.y))
                keypoints.append(2)

            label_line = [class_id, x_center, y_center, width, height] + keypoints
            label_lines.append(" ".join(map(str, label_line)))

    # ALWAYS write file (important for YOLO consistency)
    label_path = os.path.join(
        output_label_folder,
        os.path.splitext(img_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

print("Done converting to YOLO Pose format")