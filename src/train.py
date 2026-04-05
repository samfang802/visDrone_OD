import torch
from ultralytics import YOLO
import os

def train_model(config_path='config.yaml', model_type='yolo11s.pt'):
    """
    Initializes and trains the YOLO model using VisDrone dataset.
    """
    # Initialize GPU device if available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = YOLO(model_type)
    print(f"Model {model_type} loaded successfully.")

    # Define training parameters
    train_args = {
        "data": config_path,
        "epochs": 30,
        "imgsz": 768,      # Scaled to multiple of 32
        "batch": 16,       
        "device": device,
        "workers": 2,
        "project": "visdrone_detect",
        "name": "yolo11s_visdrone",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "patience": 12,
        "save": True,
        "plots": True,
        "cache": False,
        "seed": 42
    }

    # Start training
    print("Starting training phase...")
    results = model.train(**train_args)
    
    # Validate the results
    print("Starting validation phase...")
    val_results = model.val(conf=0.25, iou=0.6, augment=True, plots=True, save=True)
    
    # (Optional) Export to ONNX if needed for deployment later
    # print("Exporting model to ONNX...")
    # path = model.export(format='onnx')
    # print(f"Model exported to: {path}")

if __name__ == "__main__":
    train_model()
