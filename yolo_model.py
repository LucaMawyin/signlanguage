from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n-pose.pt")

# Train the model on the config dataset for 100 epochs
train_results = model.train(
    data="config.yaml",  # Path to dataset configuration file
    epochs=50,  # Number of training epochs
)