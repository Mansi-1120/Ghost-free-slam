import os
import cv2
import torch
from ultralytics import YOLO

# this script assumes that it is run from the DYNAMIC_SLAM directory on the SCC

def run_segmentation():
    base_dir = os.getcwd()
    dataset_root = os.path.join(base_dir, "dataset/tum_rgbd")
    output_root = os.path.join(base_dir, "segmentation/masks")
    
    # each of these should correspond to one section of the dataset
    sequences = [
        "rgbd_dataset_freiburg3_sitting_xyz",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz"
    ]

    # here we initialize the model and set to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO("yolo26n-seg.pt").to(device)

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
            
            # use YOLO26 for segmentation
            results = model.predict(img_path, conf=0.3, verbose=False)
            
            # extract masks and save them to the directory intended
            if results[0].masks is not None:
                combined_mask = torch.any(results[0].masks.data, dim=0).int() * 255
                combined_mask_np = combined_mask.cpu().numpy().astype('uint8')
                
                # filenames are saved as timestamps
                mask_filename = f"{timestamp}.png"
                cv2.imwrite(os.path.join(seq_output_dir, mask_filename), combined_mask_np)

        print(f"Done with {seq}. Masks saved to {seq_output_dir}")

if __name__ == "__main__":
    run_segmentation()