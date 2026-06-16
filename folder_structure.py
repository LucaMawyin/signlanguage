import os
import shutil

src = "yolo-data-temp/code/data/images/train"

for file in os.listdir(src):
    if not file.endswith(".jpg"):
        continue

    label = file.split("_")[0]  # B_274.jpg → B

    label_folder = os.path.join(src, label)
    os.makedirs(label_folder, exist_ok=True)

    shutil.move(
        os.path.join(src, file),
        os.path.join(label_folder, file)
    )

print("Reorganized dataset")