from ultralytics import YOLO

def main():

    model = YOLO("yolo26s.pt")

    model.train(
        data="config.yaml",
        epochs=300, 
        patience=50,
        device=0, # Nvidia 3080ti
        workers=12, # i5 13600kf 14C 20T
        batch=32,
        project=r"C:\Users\lucam\Desktop\Code\signlanguage\runs", # Explicitly state dir (saving to wrong dir otherwise) 
        name="hand_pose"
    )

    model.val(
        conf=0.25,
        project=r"C:\Users\lucam\Desktop\Code\signlanguage\runs",
        name="hand_pose_val"
    )

if __name__ == "__main__":
    main()