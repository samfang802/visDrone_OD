import os
import shutil
import cv2
import yaml


def download_data(dataset="evilspirit05/visdrone", target_dir="./data"):
    """
    Downloads VisDrone dataset from Kaggle.
    Requires kaggle API token to be set in environment.
    """
    os.makedirs(target_dir, exist_ok=True)
    os.environ['KAGGLE_API_TOKEN'] = "KGAT_9729208cd66f2a74c2fb2c7437fef861"
    
    print(f"Downloading {dataset} to {target_dir}...")
    os.system(f"kaggle datasets download -d {dataset} -p {target_dir}")
    
    zip_path = os.path.join(target_dir, "visdrone.zip")
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        # FIX: os.system("unzip") doesn't work on Windows. Using shutil instead.
        shutil.unpack_archive(zip_path, target_dir)
        os.remove(zip_path)
    else:
        print("Error: visdrone.zip not found.")

def flatten_and_setup(base_path="./data"):
    """
    Flattens redundant directory structure if present.
    """
    splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']
    for split in splits:
        outer_split_path = os.path.join(base_path, split)
        inner_split_path = os.path.join(outer_split_path, split)
        if os.path.exists(inner_split_path):
            print(f"Flattening {inner_split_path}...")
            for item in os.listdir(inner_split_path):
                shutil.move(os.path.join(inner_split_path, item), os.path.join(outer_split_path, item))
            os.rmdir(inner_split_path)
        else:
            print(f"No nested folder found for {split}, skipping.")

def convert_to_yolo(base_path="./data"):
    """
    Converts VisDrone annotations to YOLO format.
    VisDrone labels: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO labels: <object-class> <x_center> <y_center> <width> <height>
    """
    folders = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val']
    for folder_name in folders:
        img_dir = os.path.join(base_path, folder_name, 'images')
        label_dir = os.path.join(base_path, folder_name, 'annotations')
        output_dir = os.path.join(base_path, folder_name, 'labels')
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(label_dir):
            print(f"Skipping {folder_name}, annotations not found.")
            continue

        print(f"Converting {folder_name} to YOLO format...")
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue

            img_path = os.path.join(img_dir, label_file.replace('.txt', '.jpg'))
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img, _ = img.shape

            yolo_data = []
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    # FIX: Use float() first to handle cases like "100.0" which int() can't handle directly from strings
                    try:
                        x, y, w, h, score, category = [float(p) for p in parts[:6]]
                        category = int(category)
                    except ValueError:
                        continue
                    
                    # Filtering: 0 (ignored), 11 (others) are typically excluded
                    if category == 0 or category == 11:
                        continue
                    
                    # Class mapping (category 1 to 10 mapped to class 0 to 9)
                    class_id = category - 1
                    x_center = (x + w / 2) / w_img
                    y_center = (y + h / 2) / h_img
                    w_norm = w / w_img
                    h_norm = h / h_img
                    yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            with open(os.path.join(output_dir, label_file), 'w') as f_out:
                f_out.write('\n'.join(yolo_data))

    print("Success: YOLO labels generated.")

if __name__ == "__main__":
    download_data()
    flatten_and_setup()
    convert_to_yolo()
