import os
import cv2
import torch
import numpy as np
import shutil
from ultralytics import YOLO

# this script assumes that it is run from the DYNAMIC_SLAM directory on the SCC
def run_segmentation():
    dataset_root = "/projectnb/cs585/projects/dynamic_slam/dataset/tum_rgbd"
    output_root = "/projectnb/cs585/students/dhchi/course_proj/Ghost-free-slam/masking/masks"
    
    # clear existing masks
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
        print("Cleared old masks.")
    os.makedirs(output_root)

    # each of these should correspond to one section of the dataset
    sequences = [
        "rgbd_dataset_freiburg3_sitting_xyz",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz"
    ]

    # here we initialize the model and set to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO("yolo26m-seg.pt").to(device)

    for seq in sequences:
        seq_path = os.path.join(dataset_root, seq)
        rgb_txt_path = os.path.join(seq_path, "rgb.txt")
        
        # create folder for seq
        seq_output_dir = os.path.join(output_root, seq)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # else skip
        if not os.path.exists(rgb_txt_path):
            print(f"Skipping {seq}: rgb.txt not found at {rgb_txt_path}")
            continue

        print(f"Processing sequence: {seq}...")

        #parse the rgb.txt header file
        with open(rgb_txt_path, 'r') as f:
            lines = [line.split() for line in f if not line.startswith("#")]

        for timestamp, rel_img_path in lines:
            img_path = os.path.join(seq_path, rel_img_path)
            
            img = cv2.imread(img_path)
            if img is None:
                continue

            # YOLO detection
            results = model.predict(img_path, conf=0.22, iou=0.4, verbose=False, classes=[0])
            
            if results[0].masks is not None:
                # masks and bounding box gen
                masks = results[0].masks.data
                boxes = results[0].boxes.xyxy
                kernel = np.ones((10, 10), np.uint8)  # 20x20 pixel expansion — tune this
                
                for i in range(len(masks)):
                    mask_np = masks[i].cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

                    # after mask_resized = cv2.resize(...)
                    mask_dilated = cv2.dilate((mask_resized > 0.40).astype(np.uint8), kernel, iterations=2)
                    mask_smooth = cv2.GaussianBlur(mask_dilated.astype(np.float32), (21, 21), 0)

                    # apply mask
                    img[mask_smooth > 0.3] = [0, 0, 0]
                
                # save imgs to dir
                mask_filename = f"{timestamp}.png"
                cv2.imwrite(os.path.join(seq_output_dir, mask_filename), img)

        print(f"Done with {seq}. Masks saved to {seq_output_dir}")

if __name__ == "__main__":
    run_segmentation()