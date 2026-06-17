import os
import shutil

source_dir = r"C:\Users\lucam\Desktop\Code\signlanguage\yolo-data\code\data\images\train"
target_dir = r"C:\Users\lucam\Desktop\Code\signlanguage\yolo-data\code\data\images\train_flat"

os.makedirs(target_dir, exist_ok=True)

print("STARTING")
print("SOURCE:", source_dir)
print("TARGET:", target_dir)

for folder in os.listdir(source_dir):

    folder_path = os.path.join(source_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder}")
    files = os.listdir(folder_path)
    

    moved_any = False
    for i, file in enumerate(files, start=1):

        print("processing:", file)

        if not file.lower().endswith(".jpg"):
            print("skip (not jpg):", file)
            continue

        old_path = os.path.join(folder_path, file)

        new_name = f"{folder}_{i}.jpg"
        new_path = os.path.join(target_dir, new_name)

        shutil.move(old_path, new_path)

        print("moved ->", new_name)
        moved_any = True

    # delete folder only if it's empty now
    if moved_any:
        try:
            os.rmdir(folder_path)
            print("deleted folder:", folder)
        except OSError:
            print("folder not empty, skipping delete:", folder)

print("DONE")