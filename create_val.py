import os
import shutil
import random

base_path = "yolo-data-temp/code/data/images/train"
val_path = "yolo-data-temp/code/data/images/val"

split_ratio = 0.2  # 20% validation

os.makedirs(val_path, exist_ok=True)

for class_name in os.listdir(base_path):
    class_folder = os.path.join(base_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    images = os.listdir(class_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    val_images = images[:split_index]

    val_class_folder = os.path.join(val_path, class_name)
    os.makedirs(val_class_folder, exist_ok=True)

    for img in val_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(val_class_folder, img)

        shutil.move(src, dst)

print("Train/Val split complete")