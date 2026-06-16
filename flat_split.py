import os
import shutil
import random

root_dir = "yolo-data-temp"

img_train = root_dir + "/code/data/images/train"
img_val = root_dir + "/code/data/images/val"

lbl_train = root_dir + "/code/data/labels/train"
lbl_val = root_dir + "/code/data/labels/val"

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

images = [f for f in os.listdir(img_train) if f.endswith(".jpg")]
random.shuffle(images)

split = int(len(images) * 0.2)

for img in images[:split]:

    # move image
    shutil.move(
        os.path.join(img_train, img),
        os.path.join(img_val, img)
    )

    # label name
    label = img.replace(".jpg", ".txt")

    src_lbl = os.path.join(lbl_train, label)
    dst_lbl = os.path.join(lbl_val, label)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print("Done flat split")