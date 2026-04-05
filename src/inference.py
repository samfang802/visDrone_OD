import os
import pandas as pd
from ultralytics import YOLO
import argparse

def run_inference(model_path='visdrone_detect/yolo11s_visdrone/weights/best.pt', 
                  input_dir='./data/VisDrone2019-DET-val/images', 
                  output_csv='inference_results.csv'):
    """
    Runs model inference on input directory and generates a CSV report.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load trained model
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")

    # List images
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images for inference.")

    all_results = []

    # Process images
    print("Starting batch inference...")
    results = model(image_files, conf=0.25, stream=True)

    for i, result in enumerate(results):
        img_name = os.path.basename(image_files[i])
        boxes = result.boxes
        
        for box in boxes:
            # Extract box data
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            all_results.append({
                'image_id': img_name,
                'class_name': cls_name,
                'confidence': f"{conf:.4f}",
                'x1': int(coords[0]),
                'y1': int(coords[1]),
                'x2': int(coords[2]),
                'y2': int(coords[3])
            })

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisDrone Model Inference")
    parser.add_argument("--model", type=str, default='visdrone_detect/yolo11s_visdrone/weights/best.pt', help="Path to best.pt")
    parser.add_argument("--input", type=str, default='./data/VisDrone2019-DET-val/images', help="Directory of test images")
    parser.add_argument("--output", type=str, default='results.csv', help="Output CSV path")
    
    args = parser.parse_args()
    run_inference(args.model, args.input, args.output)
