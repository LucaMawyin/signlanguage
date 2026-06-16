from ultralytics import YOLO

def main():

    # Load a pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # Train the model on the config dataset for 100 epochs
    model.train(
        data="config.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        device=0, # Nvidia 3080ti
        workers=8, # 14 cores 20 threads
        project="runs/detect", # Explicitly state dir (saving to wrong dir)
    )

    model.val(conf=0.25)

if __name__ == "__main__":
    main()