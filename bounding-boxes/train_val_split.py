import os
import shutil
import random

root_dir = "yolo-data"

img_train = root_dir + "/code/data/images/train"
img_val = root_dir + "/code/data/images/val"

lbl_train = root_dir + "/code/data/labels/train"
lbl_val = root_dir + "/code/data/labels/val"

os.makedirs(img_val, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)

# support multiple formats
images = [
    f for f in os.listdir(img_train)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)

split = int(len(images) * 0.2)

val_images = images[:split]

for img in val_images:

    label = os.path.splitext(img)[0] + ".txt"

    src_img = os.path.join(img_train, img)
    dst_img = os.path.join(img_val, img)

    src_lbl = os.path.join(lbl_train, label)
    dst_lbl = os.path.join(lbl_val, label)

    # Only move if label exists AND is not empty
    if not os.path.exists(src_lbl):
        print(f"Skipping {img} (no label)")
        continue

    if os.path.getsize(src_lbl) == 0:
        print(f"Skipping {img} (empty label)")
        continue

    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)

print("Done safe split")