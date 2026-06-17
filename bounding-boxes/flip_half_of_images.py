import os
import cv2
import random

source_dir = r"C:\Users\lucam\Desktop\Code\signlanguage\yolo-data\code\data\images\train_flat"

print("STARTING IN-PLACE FLIP")
print("SOURCE:", source_dir)

files = [f for f in os.listdir(source_dir) if f.lower().endswith(".jpg")]

print("Total images:", len(files))

random.shuffle(files)

# flip half
to_flip = files[:len(files) // 2]

print("Flipping and replacing:", len(to_flip), "images")

for file in to_flip:

    path = os.path.join(source_dir, file)
    image = cv2.imread(path)

    if image is None:
        print("failed to read:", file)
        continue

    flipped = cv2.flip(image, 1)

    # overwrite original file
    cv2.imwrite(path, flipped)

    print("replaced:", file)

print("DONE")