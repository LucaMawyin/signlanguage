import os
import shutil
import random

base = "yolo-data-temp/code/data/images/train"
val = "yolo-data-temp/code/data/images/val"

os.makedirs(val, exist_ok=True)

for cls in os.listdir(base):
    folder = os.path.join(base, cls)
    if not os.path.isdir(folder):
        continue

    imgs = os.listdir(folder)
    random.shuffle(imgs)

    split = int(len(imgs) * 0.2)

    val_folder = os.path.join(val, cls)
    os.makedirs(val_folder, exist_ok=True)

    for img in imgs[:split]:
        shutil.move(
            os.path.join(folder, img),
            os.path.join(val_folder, img)
        )

print("Done splitting")